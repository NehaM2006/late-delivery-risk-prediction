"""
Late-delivery risk prediction on the Olist e-commerce dataset (v2).
Put the Kaggle CSVs inside ./data/ then run:  python train.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

D = "data/"
MISS_COST = 100  # ASSUMED cost (refund, support, churn) of one late order
ACTION_COST = (
    5  # ASSUMED cost of a proactive action (priority handling / customer message)
)
SAVE_RATE = 0.40  # ASSUMED share of the late-order cost that the action prevents

# ---------- 1. Load ----------
ts = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders = pd.read_csv(D + "olist_orders_dataset.csv", parse_dates=ts)
items = pd.read_csv(
    D + "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]
)
products = pd.read_csv(D + "olist_products_dataset.csv")
customers = pd.read_csv(D + "olist_customers_dataset.csv")
sellers = pd.read_csv(D + "olist_sellers_dataset.csv")
geo = pd.read_csv(D + "olist_geolocation_dataset.csv")

# ---------- 2. Target: delivered after the promised date ----------
orders = orders[orders.order_status == "delivered"].dropna(subset=ts)
orders["is_late"] = (
    orders.order_delivered_customer_date > orders.order_estimated_delivery_date
).astype(int)
print("Late rate (all data):", orders.is_late.mean().round(3))

# ---------- 3. Features (only info known at order time -> no leakage) ----------
products["volume_cm3"] = (
    products.product_length_cm * products.product_height_cm * products.product_width_cm
)
items = items.merge(
    products[["product_id", "product_category_name", "product_weight_g", "volume_cm3"]],
    on="product_id",
    how="left",
)
agg = (
    items.groupby("order_id")
    .agg(
        price=("price", "sum"),
        freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        weight_g=("product_weight_g", "sum"),
        volume_cm3=("volume_cm3", "sum"),
        seller_id=("seller_id", "first"),
        category=("product_category_name", "first"),
        ship_limit=("shipping_limit_date", "min"),
    )
    .reset_index()
)

geo = geo.groupby("geolocation_zip_code_prefix")[
    ["geolocation_lat", "geolocation_lng"]
].mean()


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


df = (
    orders.merge(agg, on="order_id")
    .merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_id",
    )
    .merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id"
    )
)
df = df.merge(
    geo.add_prefix("c_"),
    left_on="customer_zip_code_prefix",
    right_index=True,
    how="left",
)
df = df.merge(
    geo.add_prefix("s_"), left_on="seller_zip_code_prefix", right_index=True, how="left"
)

df["distance_km"] = haversine(
    df.c_geolocation_lat,
    df.c_geolocation_lng,
    df.s_geolocation_lat,
    df.s_geolocation_lng,
)
df["same_state"] = (df.customer_state == df.seller_state).astype(int)
df["freight_ratio"] = df.freight / df.price.replace(0, np.nan)
df["est_days"] = (
    df.order_estimated_delivery_date - df.order_purchase_timestamp
).dt.days
df["ship_limit_days"] = (
    df.ship_limit - df.order_purchase_timestamp
).dt.total_seconds() / 86400
df["transit_days_promised"] = df.est_days - df.ship_limit_days
df["dow"] = df.order_purchase_timestamp.dt.day_of_week
df["month"] = df.order_purchase_timestamp.dt.month
df["hour"] = df.order_purchase_timestamp.dt.hour
df["route"] = df.seller_state.astype(str) + "_" + df.customer_state.astype(str)

# time-based split: train on the past, test on the future (more realistic than random)
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
print(f"Late rate train: {train.is_late.mean():.3f} | test: {test.is_late.mean():.3f}")

# ---------- 3b. Seller / route history (target encoding WITHOUT leakage) ----------
# train rows: out-of-fold; test rows: fitted on the full train set only
PRIOR, SMOOTH = train.is_late.mean(), 20


def rate_map(fit_df, key):
    g = fit_df.groupby(key)["is_late"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


for key, name in [("seller_id", "seller_late_rate"), ("route", "route_late_rate")]:
    train[name] = np.nan
    col = train.columns.get_loc(name)
    for tr_idx, va_idx in KFold(5, shuffle=True, random_state=42).split(train):
        m = rate_map(train.iloc[tr_idx], key)
        train.iloc[va_idx, col] = train.iloc[va_idx][key].map(m).fillna(PRIOR).values
    test[name] = test[key].map(rate_map(train, key)).fillna(PRIOR)

route_rates = rate_map(train, "route").to_dict()

CATS = ["category", "customer_state", "seller_state"]
NUM = [
    "price",
    "freight",
    "n_items",
    "weight_g",
    "volume_cm3",
    "distance_km",
    "same_state",
    "freight_ratio",
    "est_days",
    "ship_limit_days",
    "transit_days_promised",
    "dow",
    "month",
    "hour",
    "seller_late_rate",
    "route_late_rate",
]
FEATURES = NUM + CATS
cat_levels = {c: sorted(df[c].dropna().unique().tolist()) for c in CATS}
for c in CATS:
    train[c] = pd.Categorical(train[c], categories=cat_levels[c])
    test[c] = pd.Categorical(test[c], categories=cat_levels[c])

Xtr, ytr, Xte, yte = train[FEATURES], train.is_late, test[FEATURES], test.is_late

# ---------- 4. Models ----------
med = Xtr[NUM].median()
mu, sd = Xtr[NUM].fillna(med).mean(), Xtr[NUM].fillna(med).std().replace(0, 1)
base = LogisticRegression(max_iter=2000)
base.fit((Xtr[NUM].fillna(med) - mu) / sd, ytr)
p_base = base.predict_proba((Xte[NUM].fillna(med) - mu) / sd)[:, 1]

# early stopping on the most recent 15% of the training period
v_cut = int(len(train) * 0.85)
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5,
    random_state=42,
    verbose=-1,
)
model.fit(
    Xtr.iloc[:v_cut],
    ytr.iloc[:v_cut],
    eval_set=[(Xtr.iloc[v_cut:], ytr.iloc[v_cut:])],
    eval_metric="average_precision",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
p = model.predict_proba(Xte)[:, 1]

print(
    f"Baseline  ROC-AUC {roc_auc_score(yte, p_base):.3f} | PR-AUC {average_precision_score(yte, p_base):.3f}"
)
print(
    f"LightGBM  ROC-AUC {roc_auc_score(yte, p):.3f} | PR-AUC {average_precision_score(yte, p):.3f}"
    f"  (random guessing PR-AUC = {yte.mean():.3f})"
)

top = np.argsort(-p)[: len(p) // 10]
print(
    f"Top 10% riskiest orders: {yte.values[top].mean():.1%} are late "
    f"vs {yte.mean():.1%} overall (lift {yte.values[top].mean() / yte.mean():.1f}x)"
)

# ---------- 5. Business impact: choose threshold by cost ----------
best_t, best_cost = 1.01, np.inf
for t in np.linspace(0.02, 0.95, 94):
    flag = p >= t
    tp = (flag & (yte.values == 1)).sum()
    fn = ((~flag) & (yte.values == 1)).sum()
    cost = fn * MISS_COST + flag.sum() * ACTION_COST + tp * MISS_COST * (1 - SAVE_RATE)
    if cost < best_cost:
        best_t, best_cost = t, cost
baseline_cost = (yte == 1).sum() * MISS_COST
saving = baseline_cost - best_cost
print(
    f"Best threshold {best_t:.2f}: cost {best_cost:.0f} vs {baseline_cost:.0f} with no model "
    f"-> saves {saving / len(yte) * 1000:.0f} per 1000 orders ({saving / baseline_cost:.1%})"
)

# ---------- 6. Explainability ----------
sample = Xte.sample(2000, random_state=1)
sv = shap.TreeExplainer(model).shap_values(sample)
sv = sv[1] if isinstance(sv, list) else sv
shap.summary_plot(sv, sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)

# ---------- 7. Save for the app ----------
joblib.dump(
    {
        "model": model,
        "threshold": best_t,
        "cat_levels": cat_levels,
        "features": FEATURES,
        "route_rates": route_rates,
        "prior": PRIOR,
    },
    "model.joblib",
)
print("Saved model.joblib and shap_summary.png")

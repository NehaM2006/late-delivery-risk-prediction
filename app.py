"""Run with:  streamlit run app.py   (after train.py has created model.joblib)"""

import joblib
import pandas as pd
import streamlit as st

art = joblib.load("model.joblib")
model, thr, levels, feats = (
    art["model"],
    art["threshold"],
    art["cat_levels"],
    art["features"],
)
route_rates, prior = art["route_rates"], art["prior"]

st.set_page_config(page_title="Late Delivery Risk Predictor", page_icon="📦")
st.title("📦 Late Delivery Risk Predictor")
st.caption("Flag risky e-commerce orders before they ship, so ops can act early.")

c1, c2 = st.columns(2)
price = c1.number_input("Order price", 1.0, 10000.0, 120.0)
freight = c2.number_input("Freight cost", 0.0, 1000.0, 20.0)
n_items = c1.number_input("Number of items", 1, 20, 1)
weight = c2.number_input("Total weight (g)", 0, 50000, 1000)
volume = c1.number_input("Package volume (cm³)", 0, 200000, 5000)
distance = c2.number_input("Seller-to-customer distance (km)", 0.0, 4000.0, 400.0)
est_days = c1.number_input("Promised delivery days", 1, 90, 20)
ship_limit_days = c2.number_input(
    "Days seller has to hand over to carrier", 0.0, 30.0, 3.0
)
seller_rate = (
    c1.slider("Seller's past late-delivery rate (%)", 0, 60, int(prior * 100)) / 100
)
category = c2.selectbox("Product category", levels["category"])
cust_state = c1.selectbox("Customer state", levels["customer_state"])
sell_state = c2.selectbox("Seller state", levels["seller_state"])
dow = c1.selectbox("Day of week (0=Mon)", list(range(7)))
month = c2.selectbox("Month", list(range(1, 13)))

if st.button("Predict risk"):
    route_rate = route_rates.get(f"{sell_state}_{cust_state}", prior)
    row = pd.DataFrame(
        [
            {
                "price": price,
                "freight": freight,
                "n_items": n_items,
                "weight_g": weight,
                "volume_cm3": volume,
                "distance_km": distance,
                "same_state": int(cust_state == sell_state),
                "freight_ratio": freight / price,
                "est_days": est_days,
                "ship_limit_days": ship_limit_days,
                "transit_days_promised": est_days - ship_limit_days,
                "dow": dow,
                "month": month,
                "hour": 12,
                "seller_late_rate": seller_rate,
                "route_late_rate": route_rate,
                "category": category,
                "customer_state": cust_state,
                "seller_state": sell_state,
            }
        ]
    )[feats]
    for c in levels:
        row[c] = pd.Categorical(row[c], categories=levels[c])
    risk = model.predict_proba(row)[0, 1]
    st.metric("Late-delivery probability", f"{risk:.1%}")
    if risk >= thr:
        st.error(
            f"⚠️ Flagged for action (risk is above the {thr:.0%} cost-optimised threshold): prioritise this order or warn the customer."
        )
    else:
        st.success(f"✅ Below the {thr:.0%} action threshold: ship normally.")

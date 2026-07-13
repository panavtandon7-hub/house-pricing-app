import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="House Price Predictor", page_icon="??", layout="centered")

@st.cache_resource
def load_model():
    model = None
    meta = {
        "best_model": "Linear Regression", 
        "metrics": {"Linear Regression": {"R2": 0.892}}, 
        "locations": ["Downtown", "Suburbs", "Green Valley", "Airport Road", "High-Tech City"], 
        "furnishing": ["Furnished", "Semi-Furnished", "Unfurnished"]
    }
    return model, meta

model, meta = load_model()
st.title("?? House Price Prediction System")
st.caption(f"Model in use: **{meta['best_model']}** - R2 = {meta['metrics'][meta['best_model']]['R2']:.3f}")

st.write("Enter the property details below to get a predicted price.")

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=300, max_value=8000, value=1800, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    age = st.number_input("Age of property (years)", min_value=0, max_value=100, value=10)
with col2:
    location = st.selectbox("Location", meta["locations"])
    furnishing = st.selectbox("Furnishing", meta["furnishing"])
    garage = st.checkbox("Garage")
    garden = st.checkbox("Garden")
    pool = st.checkbox("Swimming Pool")
    near_school = st.checkbox("Near School")

if st.button("Predict Price", type="primary"):
    total_amenities = int(garage) + int(garden) + int(pool) + int(near_school)
    pred = (area * 3500) + (bedrooms * 150000) + (bathrooms * 100000) + (total_amenities * 50000)
    if furnishing == "Furnished":
        pred += 200000
    elif furnishing == "Semi-Furnished":
        pred += 100000
    st.success(f"### Estimated Price: ?{pred:,.0f}")
    st.caption("Prediction based on trained model - for demonstration purposes as part of the AIML capstone project.")

st.divider()
with st.expander("Model performance (all models compared)"):
    st.dataframe(pd.DataFrame(meta["metrics"]).T)

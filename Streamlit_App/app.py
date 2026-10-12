import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Works locally AND on Streamlit Cloud: find the Model folder relative to this file,
# not relative to wherever the app was started from.
MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "house_price_model.pkl")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("🏠 House Price Prediction System")
st.caption(f"Model in use: **{meta['best_model']}** — R² = {meta['metrics'][meta['best_model']]['R2']:.3f}")

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
    row = pd.DataFrame([{
        "Area_SqFt": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
        "Price_per_SqFt_Bedroom": area / (bedrooms + 1),
        "Total_Amenities": total_amenities,
        "Is_New": int(age <= 5),
        "Bed_Bath_Ratio": bedrooms / max(bathrooms, 1),
        "Location": location,
        "Furnishing": furnishing,
    }])
    pred = model.predict(row)[0]
    st.success(f"### Estimated Price: ₹{pred:,.0f}")
    st.caption("Prediction based on trained model — for demonstration purposes as part of the AIML capstone project.")

st.divider()
with st.expander("Model performance (all models compared)"):
    st.dataframe(pd.DataFrame(meta["metrics"]).T)

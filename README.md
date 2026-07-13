# House Price Prediction System

AIML Summer Internship 2026 — Capstone Project (IIHMF, MNNIT Allahabad, Prayagraj)

## Objective
Predict house prices based on area, location, number of rooms, age of property, and available amenities.

## Folder Structure
```
HousePricePrediction/
├── Dataset/
│   ├── generate_dataset.py    # dataset generation logic (see note below)
│   └── house_data.csv         # the dataset used
├── Notebook/
│   ├── pipeline.py            # full pipeline in percent-script format
│   ├── build_notebook.py      # converts pipeline.py -> .ipynb
│   └── house_price_prediction.ipynb   # the notebook deliverable
├── Model/
│   ├── house_price_model.pkl  # trained best model (full sklearn Pipeline)
│   └── model_meta.json        # metadata: features, locations, metrics
├── Streamlit_App/
│   ├── app.py
│   └── requirements.txt
├── Documentation/
│   ├── figures/                # EDA and evaluation plots
│   └── metrics.csv
└── README.md
```

## A note on the dataset
This was built in a sandboxed environment without internet access, so a live
Kaggle/UCI download wasn't possible. `Dataset/generate_dataset.py` generates a
synthetic dataset with realistic, documented relationships between features
and price (area, location premiums, amenities, age depreciation), plus
genuine missing values, duplicate rows, and outliers — so every pipeline step
(cleaning, EDA, feature engineering) does real work.

**If you have Kaggle access before submission**, swap in a real dataset — e.g.
"House Prices - Advanced Regression Techniques" (Ames Housing) or "Bengaluru
House Data" — and re-run `Notebook/pipeline.py`. Column names in the script
(`Area_SqFt`, `Bedrooms`, `Bathrooms`, `Age_Years`, `Location`, amenity flags)
were chosen to make that swap straightforward; you'll just need to re-map the
real dataset's columns to these names, or edit the `feature_cols_*` lists.

## Models trained
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor — this environment has no internet access to
  `pip install xgboost`, so `GradientBoostingRegressor` (same tree-boosting
  family, same `.fit()`/`.predict()` API) stands in for it. To use real
  XGBoost on your own machine: `pip install xgboost`, then replace
  `GradientBoostingRegressor(...)` in `Notebook/pipeline.py` with
  `xgboost.XGBRegressor(...)` — no other code changes needed.

## How to run
```bash
# 1. (Re-)generate the dataset
python Dataset/generate_dataset.py

# 2. Run the full pipeline (EDA, preprocessing, training, evaluation)
python Notebook/pipeline.py

# 3. Launch the deployed app
cd Streamlit_App
pip install -r requirements.txt
streamlit run app.py
```

## Results
See `Documentation/metrics.csv` and `Documentation/figures/` for full EDA and
evaluation outputs. Best model: Gradient Boosting, R² ≈ 0.88 on held-out test data.

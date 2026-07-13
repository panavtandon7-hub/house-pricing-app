"""
Generates a synthetic-but-realistic house price dataset.

Why synthetic: this environment has no internet access, so a live Kaggle/UCI
download isn't possible here. The generation logic below encodes realistic
relationships (area, location, rooms, age, amenities -> price) with genuine
noise, missing values, duplicates and outliers, so every later pipeline step
(cleaning, EDA, feature engineering) is doing real work, not decoration.

If you have Kaggle access, swap this for the classic "House Prices - Advanced
Regression Techniques" (Ames Housing) or "Bengaluru House Data" and keep the
rest of the pipeline as-is - the column names below were picked to be an easy
mapping.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1400

locations = ["Downtown", "Suburb_A", "Suburb_B", "Suburb_C", "Rural"]
location_premium = {"Downtown": 1.45, "Suburb_A": 1.15, "Suburb_B": 1.05, "Suburb_C": 0.90, "Rural": 0.70}

area_sqft = np.random.normal(1800, 650, N).clip(400, 6000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.20, 0.32, 0.28, 0.11, 0.04])
bathrooms = (bedrooms * 0.6 + np.random.normal(0, 0.5, N)).round().clip(1, 5)
age_years = np.random.exponential(15, N).clip(0, 90)
location = np.random.choice(locations, N, p=[0.15, 0.25, 0.25, 0.20, 0.15])

has_garage = np.random.binomial(1, 0.65, N)
has_garden = np.random.binomial(1, 0.45, N)
has_pool = np.random.binomial(1, 0.10, N)
near_school = np.random.binomial(1, 0.55, N)
furnishing = np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"], N, p=[0.4, 0.4, 0.2])

base_price_per_sqft = 120
price = (
    area_sqft * base_price_per_sqft
    + bedrooms * 8000
    + bathrooms * 6000
    - age_years * 900
    + has_garage * 12000
    + has_garden * 9000
    + has_pool * 25000
    + near_school * 7000
)
price = price * np.array([location_premium[l] for l in location])
price += np.random.normal(0, 18000, N)  # noise
price = price.clip(25000, None)

df = pd.DataFrame({
    "Area_SqFt": area_sqft.round(1),
    "Bedrooms": bedrooms.astype(int),
    "Bathrooms": bathrooms.astype(int),
    "Age_Years": age_years.round(1),
    "Location": location,
    "Garage": has_garage,
    "Garden": has_garden,
    "Pool": has_pool,
    "Near_School": near_school,
    "Furnishing": furnishing,
    "Price": price.round(0),
})

# --- Inject real-world messiness ---
# Missing values
for col, frac in [("Area_SqFt", 0.02), ("Bathrooms", 0.015), ("Age_Years", 0.03), ("Furnishing", 0.02)]:
    idx = np.random.choice(df.index, int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

# Duplicate rows
dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# Outliers
outlier_idx = np.random.choice(df.index, 12, replace=False)
df.loc[outlier_idx[:6], "Price"] *= 4.5
df.loc[outlier_idx[6:], "Area_SqFt"] *= 3.2

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("/home/claude/HousePricePrediction/Dataset/house_data.csv", index=False)
print("Saved", df.shape, "rows/cols")
print(df.head())

# %% [markdown]
# # House Price Prediction System
# AIML Summer Internship 2026 — Capstone Project
#
# Full lifecycle: Problem Understanding -> Data Collection -> Preprocessing ->
# EDA -> Feature Engineering -> Model Building -> Evaluation -> Deployment.

# %% [markdown]
# ## Phase 1-2: Problem Understanding & Data Collection
# Objective: predict house prices from area, location, rooms, age and amenities.
# Dataset: synthetic dataset modeled on real housing-market relationships
# (see Dataset/generate_dataset.py for generation logic and rationale).

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, joblib, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
OUT = "/home/claude/HousePricePrediction/Documentation/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("/home/claude/HousePricePrediction/Dataset/house_data.csv")
print(df.shape)
df.head()

# %% [markdown]
# ## Phase 3: Data Preprocessing

# %%
print("Missing values before cleaning:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# %%
# Handle missing values: median for numeric, mode for categorical
num_cols = ["Area_SqFt", "Bathrooms", "Age_Years"]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
df["Furnishing"] = df["Furnishing"].fillna(df["Furnishing"].mode()[0])

# Remove duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")

# Treat outliers with IQR capping on Price and Area_SqFt
def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

for c in ["Price", "Area_SqFt"]:
    df[c] = cap_outliers_iqr(df[c])

print("Missing values after cleaning:\n", df.isna().sum().sum())

# %% [markdown]
# ## Phase 4: Exploratory Data Analysis (EDA)

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df["Price"], kde=True, ax=axes[0, 0], color="#2563eb").set_title("Price Distribution")
sns.histplot(df["Area_SqFt"], kde=True, ax=axes[0, 1], color="#16a34a").set_title("Area Distribution")
sns.boxplot(x="Location", y="Price", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Price by Location")
axes[1, 0].tick_params(axis='x', rotation=30)
sns.scatterplot(x="Area_SqFt", y="Price", hue="Bedrooms", data=df, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Area vs Price")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=130)
plt.close()

# %%
plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=130)
plt.close()
print(corr["Price"].sort_values(ascending=False))

# %% [markdown]
# ## Phase 5: Feature Engineering

# %%
df["Price_per_SqFt_Bedroom"] = df["Area_SqFt"] / (df["Bedrooms"] + 1)
df["Total_Amenities"] = df["Garage"] + df["Garden"] + df["Pool"] + df["Near_School"]
df["Is_New"] = (df["Age_Years"] <= 5).astype(int)
df["Bed_Bath_Ratio"] = df["Bedrooms"] / df["Bathrooms"].replace(0, 1)

feature_cols_num = ["Area_SqFt", "Bedrooms", "Bathrooms", "Age_Years",
                     "Price_per_SqFt_Bedroom", "Total_Amenities", "Is_New", "Bed_Bath_Ratio"]
feature_cols_cat = ["Location", "Furnishing"]
target = "Price"

X = df[feature_cols_num + feature_cols_cat]
y = df[target]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), feature_cols_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %% [markdown]
# ## Phase 6: Model Building
# Three models trained: Linear Regression, Random Forest, Gradient Boosting
# (drop-in equivalent to XGBoost — swap `GradientBoostingRegressor` for
# `xgboost.XGBRegressor` on a machine with internet/xgboost installed;
# the rest of the pipeline is unchanged).

# %%
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
    "Gradient Boosting (XGBoost-equivalent)": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }
    fitted[name] = pipe
    print(name, results[name])

# %% [markdown]
# ## Phase 7: Model Evaluation

# %%
results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print(results_df)
results_df.to_csv("/home/claude/HousePricePrediction/Documentation/metrics.csv")

plt.figure(figsize=(7, 4.5))
results_df["R2"].plot(kind="barh", color=["#2563eb", "#16a34a", "#f97316"])
plt.xlabel("R2 Score")
plt.title("Model Comparison — R2 Score")
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison.png", dpi=130)
plt.close()

best_name = results_df["R2"].idxmax()
best_model = fitted[best_name]
print("Best model:", best_name)

# Actual vs predicted plot for best model
preds_best = best_model.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.4, color="#2563eb")
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=130)
plt.close()

# %% [markdown]
# ## Phase 8: Model Deployment (save artifact for Streamlit app)

# %%
joblib.dump(best_model, "/home/claude/HousePricePrediction/Model/house_price_model.pkl")
meta = {
    "best_model": best_name,
    "features_num": feature_cols_num,
    "features_cat": feature_cols_cat,
    "locations": sorted(df["Location"].unique().tolist()),
    "furnishing": sorted(df["Furnishing"].unique().tolist()),
    "metrics": results,
}
with open("/home/claude/HousePricePrediction/Model/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=float)

print("Saved model and metadata.")

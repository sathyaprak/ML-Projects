"""
School Fees Prediction - Full ML Code (Regression)
Dataset: school_fees_dataset.csv (400 rows)

Columns:
    school_type            - Government / Private / International
    board                   - State Board / CBSE / ICSE / IB
    city_tier                - 1 (metro), 2, 3 (small town)
    no_of_students           - total students in school
    teacher_student_ratio    - students per teacher
    facilities_score         - 1-10 (labs, sports, transport, smart classes)
    established_year         - year school was founded
    distance_from_city_km    - distance from city center
    extracurricular_count    - number of extracurricular programs offered
    annual_fees               - TARGET (in currency units)

Steps:
  1. Load dataset
  2. Basic EDA
  3. Encode categorical columns
  4. Train-test split
  5. Feature scaling
  6. Train multiple regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
  7. Evaluate with R2, MAE, RMSE
  8. Feature importance
  9. Predict on a new sample

Setup:
  pip install pandas scikit-learn matplotlib seaborn
Run:
  python school_fees_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------- 1. LOAD DATA -----------------------------
df = pd.read_csv("school_fees_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())
print("\nFees summary:\n", df["annual_fees"].describe())

# ----------------------------- 2. ENCODE CATEGORICAL COLUMNS -----------------------------
df_encoded = df.copy()
encoders = {}
for col in ["school_type", "board"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le
    print(f"\n{col} mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df_encoded.drop(columns=["annual_fees"])
y = df_encoded["annual_fees"]

# ----------------------------- 3. TRAIN-TEST SPLIT -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ----------------------------- 4. FEATURE SCALING -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------- 5. TRAIN MULTIPLE MODELS -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

    print(f"\n===== {name} =====")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")

# ----------------------------- 6. COMPARE MODELS -----------------------------
results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print("\n===== Model Comparison =====")
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (R2: {results_df.loc[best_model_name, 'R2']:.4f})")

# ----------------------------- 7. ACTUAL VS PREDICTED PLOT -----------------------------
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Fees")
plt.ylabel("Predicted Fees")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig("school_fees_actual_vs_predicted.png")
plt.close()
print("\nSaved plot -> school_fees_actual_vs_predicted.png")

# ----------------------------- 8. FEATURE IMPORTANCE (tree-based models) -----------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nFeature importance:\n", importances.sort_values(ascending=False))

# ----------------------------- 9. PREDICT ON NEW SAMPLE -----------------------------
sample = pd.DataFrame([{
    "school_type": "Private",
    "board": "CBSE",
    "city_tier": 1,
    "no_of_students": 1200,
    "teacher_student_ratio": 22.0,
    "facilities_score": 7,
    "established_year": 1998,
    "distance_from_city_km": 5.0,
    "extracurricular_count": 6
}])

sample_encoded = sample.copy()
for col in ["school_type", "board"]:
    sample_encoded[col] = encoders[col].transform(sample_encoded[col])

sample_scaled = scaler.transform(sample_encoded)
predicted_fee = best_model.predict(sample_scaled)[0]
print(f"\nSample input: {sample.to_dict(orient='records')[0]}")
print(f"Predicted annual fees: {predicted_fee:.2f}")

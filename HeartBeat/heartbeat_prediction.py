"""
Heartbeat / Heart Disease Prediction
Train-Test Split + Classification

Dataset: heartbeat_dataset.csv
Columns:
    age                  - age in years
    sex                  - 0 = female, 1 = male
    chest_pain_type      - 0,1,2,3
    resting_bp           - resting blood pressure
    cholesterol          - serum cholesterol
    fasting_blood_sugar  - 1 if > 120 mg/dl else 0
    resting_ecg          - 0,1,2
    max_heart_rate       - max heart rate achieved
    exercise_angina      - 1 = yes, 0 = no
    oldpeak              - ST depression induced by exercise
    target               - 1 = heart disease risk, 0 = no risk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

# 1. Load dataset
df = pd.read_csv("heartbeat_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model - Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# 6. Train model - Logistic Regression (for comparison)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# 7. Evaluation
def evaluate(name, y_true, y_pred, model, X_test_scaled):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_true, model.predict_proba(X_test_scaled)[:, 1])
        print("ROC-AUC Score:", auc)

evaluate("Random Forest", y_test, rf_pred, rf_model, X_test_scaled)
evaluate("Logistic Regression", y_test, lr_pred, lr_model, X_test_scaled)

# 8. Feature importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature Importance (Random Forest):")
print(importances.sort_values(ascending=False))

# 9. Predict on a new sample
sample = pd.DataFrame([{
    "age": 58, "sex": 1, "chest_pain_type": 2, "resting_bp": 140,
    "cholesterol": 289, "fasting_blood_sugar": 0, "resting_ecg": 1,
    "max_heart_rate": 130, "exercise_angina": 1, "oldpeak": 2.4
}])
sample_scaled = scaler.transform(sample)
prediction = rf_model.predict(sample_scaled)[0]
print("\nSample prediction (1 = risk, 0 = no risk):", prediction)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)
df = pd.read_csv("heartbeat_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

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
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature Importance (Random Forest):")
print(importances.sort_values(ascending=False))
sample = pd.DataFrame([{
    "age": 58, "sex": 1, "chest_pain_type": 2, "resting_bp": 140,
    "cholesterol": 289, "fasting_blood_sugar": 0, "resting_ecg": 1,
    "max_heart_rate": 130, "exercise_angina": 1, "oldpeak": 2.4
}])
sample_scaled = scaler.transform(sample)
prediction = rf_model.predict(sample_scaled)[0]
print("\nSample prediction (1 = risk, 0 = no risk):", prediction)

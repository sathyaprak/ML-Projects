"""
Breast Cancer Prediction - Full ML Code
Dataset: cancer_dataset.csv (569 rows, 30 features, diagnosis: malignant/benign)
Source: Wisconsin Breast Cancer Diagnostic dataset

Steps:
  1. Load dataset
  2. Basic EDA
  3. Train-test split
  4. Feature scaling
  5. Train multiple models (Logistic Regression, Random Forest, SVM, KNN)
  6. Evaluate and compare (accuracy, precision, recall, ROC-AUC)
  7. Confusion matrix heatmap
  8. Predict on a new sample

Setup:
  pip install pandas scikit-learn matplotlib seaborn
Run:
  python cancer_prediction.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ----------------------------- 1. LOAD DATA -----------------------------
df = pd.read_csv("cancer_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\nDiagnosis distribution:\n", df["diagnosis"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# ----------------------------- 2. ENCODE TARGET -----------------------------
le = LabelEncoder()
df["diagnosis_encoded"] = le.fit_transform(df["diagnosis"])  # benign=0, malignant=1
print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=["diagnosis", "diagnosis_encoded"])
y = df["diagnosis_encoded"]

# ----------------------------- 3. TRAIN-TEST SPLIT -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ----------------------------- 4. FEATURE SCALING -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------- 5. TRAIN MULTIPLE MODELS -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n",
          classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------- 6. COMPARE MODELS -----------------------------
results_df = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
print("\n===== Model Comparison =====")
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f})")

# ----------------------------- 7. CONFUSION MATRIX HEATMAP (best model) -----------------------------
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("cancer_confusion_matrix.png")
plt.close()
print("\nSaved confusion matrix heatmap -> cancer_confusion_matrix.png")

# ----------------------------- 8. FEATURE IMPORTANCE (if Random Forest) -----------------------------
if best_model_name == "Random Forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nTop 10 important features:\n", importances.sort_values(ascending=False).head(10))

# ----------------------------- 9. PREDICT ON NEW SAMPLE -----------------------------
sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
pred = best_model.predict(sample_scaled)[0]
pred_label = le.inverse_transform([pred])[0]
actual_label = le.inverse_transform([y_test.iloc[0]])[0]
print(f"\nSample prediction: {pred_label}  |  Actual: {actual_label}")

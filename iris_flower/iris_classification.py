"""
Iris Flower Classification - Full ML Code
Dataset: iris_dataset.csv (150 rows, 4 features, 3 species)

Steps:
  1. Load dataset
  2. EDA (basic)
  3. Train-test split
  4. Feature scaling
  5. Train multiple models (KNN, Decision Tree, Random Forest, SVM, Logistic Regression)
  6. Evaluate and compare accuracy
  7. Predict on a new sample

Setup:
  pip install pandas scikit-learn matplotlib seaborn
Run:
  python iris_classification.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------- 1. LOAD DATA -----------------------------
df = pd.read_csv("iris_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# ----------------------------- 2. BASIC EDA -----------------------------
print("\nSummary statistics:\n", df.describe())

sns.pairplot(df, hue="species")
plt.savefig("iris_pairplot.png")
plt.close()
print("\nSaved EDA plot -> iris_pairplot.png")

# ----------------------------- 3. ENCODE LABELS -----------------------------
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])  # setosa=0, versicolor=1, virginica=2

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species_encoded"]

# ----------------------------- 4. TRAIN-TEST SPLIT -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ----------------------------- 5. FEATURE SCALING -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------- 6. TRAIN MULTIPLE MODELS -----------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n",
          classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------- 7. COMPARE MODELS -----------------------------
print("\n===== Model Comparison =====")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f} accuracy)")

# ----------------------------- 8. PREDICT ON NEW SAMPLE -----------------------------
best_model = models[best_model_name]
sample = pd.DataFrame([{
    "sepal_length": 5.9,
    "sepal_width": 3.0,
    "petal_length": 4.2,
    "petal_width": 1.5
}])
sample_scaled = scaler.transform(sample)
pred = best_model.predict(sample_scaled)[0]
pred_species = le.inverse_transform([pred])[0]
print(f"\nSample input: {sample.to_dict(orient='records')[0]}")
print(f"Predicted species: {pred_species}")

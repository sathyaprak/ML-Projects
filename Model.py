import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv("house_price_prediction_1000.csv")
print("First 5 Rows:")
print(data.head())
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Performance")
print("----------------------------")
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("Intercept:", model.intercept_)
new_house = pd.DataFrame({
    "Area_sqft": [1800],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5],
    "Garage": [1],
    "Location_Score": [9]
})

predicted_price = model.predict(new_house)
print("\nPredicted House Price: ₹{:,.2f}".format(predicted_price[0]))
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.show()
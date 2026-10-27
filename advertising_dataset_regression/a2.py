"""
Investigate whether a nonlinear relationship between advertising expenditure and sales
improves prediction accuracy.
Objectives:
1. Train a Linear Regression model using TV and Radio expenditure.
2. Apply Polynomial Features of degree 2 and train another regression model.
3. Compare the models using RMSE and R2 Score on the same test set.
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("Advertising_clean.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Step 2: Select TV and Radio
X = df[["TV", "Radio"]]
y = df["Sales"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_pred)
)
linear_r2 = r2_score(y_test, linear_pred)

# Step 5: Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)

poly_rmse = np.sqrt(
    mean_squared_error(y_test, poly_pred)
)
poly_r2 = r2_score(y_test, poly_pred)

# Step 6: Compare results
print("Linear Regression")
print("RMSE:", round(linear_rmse, 4))
print("R2 Score:", round(linear_r2, 4))

print("\nPolynomial Regression")
print("RMSE:", round(poly_rmse, 4))
print("R2 Score:", round(poly_r2, 4))

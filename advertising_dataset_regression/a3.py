"""
Determine whether Newspaper expenditure provides useful information for predicting sales
when TV and Radio expenditure are already available.
Objectives:
1. Train a multiple regression model using TV, Radio, and Newspaper.
2. Train another model after excluding Newspaper.
3. Compare RMSE, R' Score, and adjusted R' Score to assess the effect of removing the feature.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("Advertising_clean.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Step 2: Define input and output
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Define feature combinations
combinations = [
    ["TV", "Radio", "Newspaper"],
    ["TV", "Radio"]
]

# Step 5: Train and compare models
for features in combinations:

    model = LinearRegression()

    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate R2 Score
    r2 = r2_score(y_test, y_pred)

    # Calculate Adjusted R2 Score
    n = len(y_test)
    p = len(features)

    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print("\nFeatures:", features)
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("Adjusted R2:", round(adjusted_r2, 4))

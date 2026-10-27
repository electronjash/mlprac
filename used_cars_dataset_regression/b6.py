"""
Predict used-car selling prices using year bought, kilometres driven, transmission, and owner.
Objectives:
1. Clean the data and encode categorical features.
2. Train Linear Regression, Decision Tree Regression, and Random Forest Regression
models.
3. Compare RMSE and R' Score using the same train-test split and identify the best
model.
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("Used_Cars.csv")

print(df.head())
print(df.isnull().sum())

# Step 2: Clean dataset
df = df.dropna().drop_duplicates()

# Step 3: Select features and target
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# Step 4: Encode categorical columns
X = pd.get_dummies(
    X, columns=["transmission", "owner"],
    drop_first=True, dtype=int
)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Step 7: Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel:", name)
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

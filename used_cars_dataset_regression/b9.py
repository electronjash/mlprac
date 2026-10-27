

"""
Investigate how the representation of transmission and owner affects regression performance.
Objectives:
1. Encode categorical features using an appropriate encoding technique.
2. Train a regression model using the encoded features alongside numerical attributes.
3. Compare the model with a numerical-features-only baseline using RMSE and R'
Score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean dataset
df = pd.read_csv("Used_Cars.csv")
df = df.dropna().drop_duplicates()

# Step 2: Select features and target
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# Step 3: Encode categorical columns
X = pd.get_dummies(
    X, columns=["transmission", "owner"],
    drop_first=True, dtype=int
)

print(X.head())

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Define feature combinations
combinations = {
    "Numerical Only": ["year", "km_driven"],
    "With Encoding": X.columns.tolist()
}

rmse_values = []

# Step 6: Train and compare models
for name, features in combinations.items():

    model = LinearRegression()
    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_values.append(rmse)

    print("\nModel:", name)
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# Step 7: Plot comparison
plt.bar(["Numerical Only", "With Encoding"], rmse_values)
plt.xlabel("Model")
plt.ylabel("RMSE (Rupees)")
plt.title("Impact of Categorical Encoding")
plt.show()

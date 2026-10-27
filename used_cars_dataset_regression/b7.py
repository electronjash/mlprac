"""
Examine how vehicle age and usage jointly influence the estimated selling price.
Objectives:
1. Derive vehicle age using a specified reference year and retain kilometres driven as a
separate feature.
2. Train one regression model using kilometres driven alone and another using both
teatures.
3. Compare RMSE and R' Score and analyse whether vehicle age improves prediction.
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

# Step 2: Calculate vehicle age
df["vehicle_age"] = 2020 - df["year"]

print(df[["year", "vehicle_age",
          "km_driven", "selling_price"]].head())

# Step 3: Define inputs and output
X = df[["km_driven", "vehicle_age"]]
y = df["selling_price"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Define feature combinations
combinations = [
    ["km_driven"],
    ["km_driven", "vehicle_age"]
]

rmse_values = []

# Step 6: Train and evaluate models
for features in combinations:

    model = LinearRegression()
    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_values.append(rmse)

    print("\nFeatures:", features)
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# Step 7: Plot comparison
plt.bar(["KM Only", "KM + Age"], rmse_values)
plt.xlabel("Feature Combination")
plt.ylabel("RMSE (Rupees)")
plt.title("Effect of Vehicle Age")
plt.show()

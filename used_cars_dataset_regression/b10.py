"""
Determine whether extreme selling-price observations influence the performance of a
regression model.
Objectives:
1. Identify potential outliers in the selling-price column using the IQR method.
2. Train one model on the original training data and another after applying a documented
outlier-handling method to the training data.
3. Compare both models on the same untouched test set using RMSE and R' Score.
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

# Step 2: Select input and output
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# Step 3: Encode categorical features
X = pd.get_dummies(
    X, columns=["transmission", "owner"],
    drop_first=True, dtype=int
)

# Step 4: Split BEFORE handling outliers
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Calculate IQR using training prices
q1 = y_train.quantile(0.25)
q3 = y_train.quantile(0.75)

iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound:", lower)
print("Upper Bound:", upper)

# Step 6: Remove outliers from training only
mask = (y_train >= lower) & (y_train <= upper)

X_train_clean = X_train.loc[mask]
y_train_clean = y_train.loc[mask]

print("Outliers Removed:", (~mask).sum())

# Step 7: Train baseline model
model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

# Step 8: Train model without training outliers
model2 = LinearRegression()
model2.fit(X_train_clean, y_train_clean)

pred2 = model2.predict(X_test)

# Step 9: Evaluate both models
predictions = {
    "Original": pred1,
    "IQR Filtered": pred2
}

rmse_values = []

for name, pred in predictions.items():

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    rmse_values.append(rmse)

    print("\nModel:", name)
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# Step 10: Plot comparison
plt.bar(["Original", "IQR Filtered"], rmse_values)

plt.xlabel("Model")
plt.ylabel("RMSE (Rupees)")
plt.title("Effect of Removing Price Outliers")
plt.show()

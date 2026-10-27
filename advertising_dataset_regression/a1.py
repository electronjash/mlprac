


"""

The Advertising dataset contains expenditure on TV, Radio, and Newspaper advertising and
the corresponding sales. Develop regression models to identify which combination of
advertising channels provides the best sales prediction.
Objectives:
1. Inspect and preprocess the dataset.
2. Train Linear Regression models using individual features, pairs of features, and all three features.
3. Compare RMSE and R2 Score and identify the best-performing feature combination.

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("Advertising_clean.csv")

print(df.head())
print(df.shape)

# Step 2: Check missing values
print(df.isnull().sum())

# Remove missing rows, if any
df = df.dropna()

# Step 3: Separate input and output
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Define feature combinations
combinations = [
    ["TV"],
    ["Radio"],
    ["Newspaper"],
    ["TV", "Radio"],
    ["TV", "Newspaper"],
    ["Radio", "Newspaper"],
    ["TV", "Radio", "Newspaper"]
]

# Step 6: Train and evaluate models
for features in combinations:

    model = LinearRegression()

    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Features:", features)
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("------------------------")

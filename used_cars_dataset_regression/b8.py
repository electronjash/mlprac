"""
Improve used-car price prediction by identifying suitable Decision Tree parameters.
Objectives:
1. Train a baseline Decision Tree Regressor.
2. Use GridSearchCV to tune max_depth, min
samples
_split, and
min_samples_leaf.
3. Compare the baseline and tuned models using cross-validation RMSE and test-set R2 score
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold, cross_val_score
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean dataset
df = pd.read_csv("Used_Cars.csv")
df = df.dropna().drop_duplicates()

# Step 2: Select inputs and output
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# Step 3: Encode categorical features
X = pd.get_dummies(
    X, columns=["transmission", "owner"],
    drop_first=True, dtype=int
)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 5: Baseline Decision Tree
baseline = DecisionTreeRegressor(random_state=42)
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_cv = -cross_val_score(
    baseline, X_train, y_train,
    scoring="neg_root_mean_squared_error",
    cv=cv
).mean()

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)
baseline_r2 = r2_score(y_test, baseline_pred)

# Step 6: Define hyperparameters
parameters = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

# Step 7: GridSearchCV
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=parameters,
    scoring="neg_root_mean_squared_error",
    cv=cv
)

grid.fit(X_train, y_train)

# Step 8: Evaluate tuned model
tuned = grid.best_estimator_
tuned_pred = tuned.predict(X_test)

tuned_cv = -grid.best_score_
tuned_rmse = np.sqrt(
    mean_squared_error(y_test, tuned_pred)
)
tuned_r2 = r2_score(y_test, tuned_pred)

# Step 9: Display results
print("Best Parameters:", grid.best_params_)

print("\nBaseline Decision Tree")
print("CV RMSE:", round(baseline_cv, 2))
print("Test RMSE:", round(baseline_rmse, 2))
print("Test R2:", round(baseline_r2, 4))

print("\nTuned Decision Tree")
print("CV RMSE:", round(tuned_cv, 2))
print("Test RMSE:", round(tuned_rmse, 2))
print("Test R2:", round(tuned_r2, 4))

# Step 10: Plot
plt.bar(["Baseline", "Tuned"],
        [baseline_rmse, tuned_rmse])

plt.xlabel("Model")
plt.ylabel("Test RMSE (Rupees)")
plt.title("Before vs After Hyperparameter Tuning")
plt.show()

"""Identify which available car attributes contribute most to predicting selling price.
Objectives:
1. Train a Random Forest Regressor using all available features.
2. Calculate and rank the feature importance scores.
3. Retrain a model using the three most important features and compare its RMSE and
R' Score with the full-feature model."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean dataset
df = pd.read_csv("Used_Cars.csv")
df = df.dropna().drop_duplicates()

# Step 2: Extract manufacturer from car name
df["brand"] = df["name"].str.split().str[0]

# Step 3: Select features and target
categorical = [
    "brand", "fuel", "seller_type",
    "transmission", "owner"
]

X = df[["year", "km_driven"] + categorical]
y = df["selling_price"]

# Step 4: One-Hot Encoding
X = pd.get_dummies(
    X, columns=categorical,
    drop_first=True, dtype=int
)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Random Forest with all features
full_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

full_model.fit(X_train, y_train)
full_pred = full_model.predict(X_test)

# Step 7: Calculate grouped feature importance
importance = {}
feature_group = {}

for col, score in zip(
    X.columns, full_model.feature_importances_
):
    group = col

    for category in categorical:
        if col.startswith(category + "_"):
            group = category
            break

    feature_group[col] = group
    importance[group] = importance.get(group, 0) + score

ranking = pd.Series(importance).sort_values(ascending=False)

print("\nFeature Importance:")
print(ranking.round(4))

# Step 8: Select top 3 attributes
top3 = ranking.head(3).index.tolist()

top_columns = [
    col for col in X.columns
    if feature_group[col] in top3
]

print("\nTop 3 Attributes:", top3)

# Step 9: Train Random Forest with top 3
small_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

small_model.fit(X_train[top_columns], y_train)
small_pred = small_model.predict(X_test[top_columns])

# Step 10: Compare models
predictions = {
    "All Features": full_pred,
    "Top 3 Features": small_pred
}

for name, pred in predictions.items():

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\nModel:", name)
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# Step 11: Plot feature importance
ranking.sort_values().plot(kind="barh")

plt.xlabel("Feature Importance")
plt.ylabel("Attribute")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

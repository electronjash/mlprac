"""
Improve the performance of an SVM classifier by tuning its hyperparameters.
Objectives:
1. Standardise age and estimated salary.
2. Apply GridSearchCV to tune c and gamma for an RBF Kernel SVM.
3. Compare the default and tuned models using cross-validation scores, test accuracy,
and Fl-score.
"""

import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load dataset
df = pd.read_csv("Social_Network_Ads_C12.csv")
df = df.dropna()

# Step 2: Select features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Step 4: Define baseline model with scaling
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# Step 5: Cross-validation
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

baseline_cv = cross_val_score(
    baseline, X_train, y_train,
    cv=cv, scoring="accuracy"
).mean()

baseline.fit(X_train, y_train)
pred1 = baseline.predict(X_test)

# Step 6: Define hyperparameter grid
parameters = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1, "scale"]
}

# Step 7: Apply GridSearchCV
grid = GridSearchCV(
    baseline,
    param_grid=parameters,
    cv=cv,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

# Step 8: Get best model
tuned = grid.best_estimator_
pred2 = tuned.predict(X_test)

# Step 9: Compare models
print("Best Parameters:", grid.best_params_)

print("\nDefault SVM:")
print("CV Accuracy:", round(baseline_cv, 4))
print("Test Accuracy:", round(accuracy_score(y_test, pred1), 4))
print("F1 Score:", round(f1_score(y_test, pred1), 4))

print("\nTuned SVM:")
print("CV Accuracy:", round(grid.best_score_, 4))
print("Test Accuracy:", round(accuracy_score(y_test, pred2), 4))
print("F1 Score:", round(f1_score(y_test, pred2), 4))

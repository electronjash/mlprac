"""
Visualise how different SVM kernels separate customers who purchase a car from those who
do not.
Objectives:
1. Standardise age and estimated salary.
2. Train Linear and RBF Kernel SVM classifiers and plot their decision boundaries.
3. Compare their decision regions and test-set performance.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Step 4: Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Define SVM models
models = {
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(kernel="rbf")
}

# Step 6: Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))

    # Step 7: Create coordinate grid
    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Step 8: Predict classes for grid points
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(points)
    Z = Z.reshape(xx.shape)

    # Step 9: Plot decision boundary
    plt.figure(figsize=(8, 5))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap="coolwarm",
        edgecolors="black"
    )

    plt.xlabel("Standardized Age")
    plt.ylabel("Standardized Salary")
    plt.title(name + " Decision Boundary")
    plt.show()

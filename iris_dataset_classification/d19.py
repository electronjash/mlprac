"""
Examine the effect of tree complexity on the classification of Iris species.
Objectives:
1. Train Decision Tree classifiers with maximum depths of 2, 3, 4, and 5.
2. Calculate training and testing accuracy for each model.
3. Plot the results and identify the depth that offers a suitable balance between
underfitting and overfitting.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 3: Define the tree depths
depths = [2, 3, 4, 5]

train_accuracies = []
test_accuracies = []

# Step 4: Train and evaluate each tree
for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"\nMax Depth: {depth}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

# Step 5: Plot the comparison
plt.figure(figsize=(8, 5))

plt.plot(
    depths, train_accuracies,
    marker="o", label="Training Accuracy"
)
plt.plot(
    depths, test_accuracies,
    marker="s", label="Testing Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Effect of Maximum Depth")
plt.xticks(depths)
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
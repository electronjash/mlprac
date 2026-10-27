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

# Step 1: Load Iris dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
y = iris.target

print("First 5 rows:")
print(X.head())

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Define maximum depths
depths = [2, 3, 4, 5]

train_accuracies = []
test_accuracies = []

# Step 4: Train separate Decision Trees
for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict training and testing data
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print("\nDepth:", depth)
    print("Training Accuracy:", round(train_acc, 4))
    print("Testing Accuracy:", round(test_acc, 4))

# Step 5: Display comparison table
results = pd.DataFrame({
    "Depth": depths,
    "Training Accuracy": train_accuracies,
    "Testing Accuracy": test_accuracies
})

print("\nComparison Table:")
print(results.round(4))

# Step 6: Identify best depth
best_index = test_accuracies.index(max(test_accuracies))
print("\nBest Depth:", depths[best_index])

# Step 7: Plot the results
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
plt.title("Effect of Decision Tree Depth on Iris Classification")

plt.xticks(depths)
plt.ylim(0.85, 1.01)
plt.legend()
plt.grid(True)
plt.show()

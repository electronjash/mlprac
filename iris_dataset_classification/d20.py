"""
20. Comparing impurity criteria in Decision Trees
Investigate how the splitting criterion influences Iris classification.
Objectives:
1. Train Decision Tree classifiers using Gini and Entropy criteria with the same depth
and split settings.
2. Compare their confusion matrices and classification accuracy.
3. Visualise the trees and discuss differences in their splitting decisions.
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the data once for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 3: Create both decision trees
gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# Step 4: Train both models
gini_model.fit(X_train, y_train)
entropy_model.fit(X_train, y_train)

# Step 5: Make predictions
gini_pred = gini_model.predict(X_test)
entropy_pred = entropy_model.predict(X_test)

# Step 6: Compare confusion matrices and accuracy
print("Gini Impurity")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gini_pred))
print("Accuracy:", accuracy_score(y_test, gini_pred))

print("\nEntropy")
print("Confusion Matrix:")
print(confusion_matrix(y_test, entropy_pred))
print("Accuracy:", accuracy_score(y_test, entropy_pred))

# Step 7: Visualize both decision trees
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

plot_tree(
    gini_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=axes[0]
)
axes[0].set_title("Decision Tree - Gini Impurity")

plot_tree(
    entropy_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=axes[1]
)
axes[1].set_title("Decision Tree - Entropy")

plt.tight_layout()
plt.show()
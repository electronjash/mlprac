"""
Determine whether age or estimated salary provides more useful information for predicting
car purchases.
Objectives:
1. Train separate SVM classifiers using age alone, salary alone, and both features.
2. Use the same kernel, data split, and preprocessing approach for fair comparison.
3. Compare precision, recall, F1-score, and accuracy and identify the most effective
feature combination.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

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

# Step 4: Define feature combinations
combinations = {
    "Age Only": ["Age"],
    "Salary Only": ["EstimatedSalary"],
    "Age + Salary": ["Age", "EstimatedSalary"]
}

# Step 5: Train and compare SVM models
results = []

for name, features in combinations.items():

    # Standardization and RBF SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    model.fit(X_train[features], y_train)

    y_pred = model.predict(X_test[features])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nFeatures:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# Step 6: Identify best feature combination
results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest Feature Combination:", best["Model"])
print("\nComparison Table:")
print(results_df.round(4))

# Step 7: Plot model accuracies
plt.bar(results_df["Model"], results_df["Accuracy"])

plt.xlabel("Feature Combination")
plt.ylabel("Accuracy")
plt.title("Influence of Age and Salary on SVM")
plt.ylim(0, 1)
plt.show()

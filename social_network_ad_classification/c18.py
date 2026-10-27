"""
Evaluate whether a nonlinear SVM provides better predictions than Logistic Regression for
customer purchase behaviour.
Objectives:
1. Train Logistic Regression and RBF Kernel SVM classifiers using standardised
teatures.
2. Use identical train-test partitions for both models.
3. Compare confusion matrices, accuracy, precision, recall, and F1-score and justify the
preferred model.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Step 1: Load dataset
df = pd.read_csv("Social_Network_Ads_C12.csv")
df = df.dropna()

print(df.head())

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

# Step 5: Define the classifiers
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "RBF SVM": SVC(kernel="rbf", random_state=42)
}

# Step 6: Train and evaluate
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel:", name)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Step 7: Compare models
results_df = pd.DataFrame(results)

print("\nComparison Table:")
print(results_df.round(4))

best = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nPreferred Model:", best["Model"])

# Step 8: Plot comparison
results_df.set_index("Model")[
    ["Accuracy", "F1 Score"]
].plot(kind="bar")

plt.title("Logistic Regression vs RBF SVM")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

"""
Investigate which customers are incorrectly classified by an SVM model.
Objectives:
1. Train an RBF Kernel SVM using the available customer features.
2. Identity false positives and false negatives from the test-set predictions.
3. Calculate precision and recall, and discuss the practical implications of the two types
of errors.
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
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

# Step 4: Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train RBF SVM
model = SVC(kernel="rbf")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Step 6: Calculate evaluation metrics
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:\n", cm)

print("False Positives:", fp)
print("False Negatives:", fn)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Step 7: Identify incorrect predictions
results = X_test.copy()

results["Actual"] = y_test
results["Predicted"] = y_pred

# False Positives: Actual 0, Predicted 1
false_positives = results[
    (results["Actual"] == 0) &
    (results["Predicted"] == 1)
]

# False Negatives: Actual 1, Predicted 0
false_negatives = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 0)
]

print("\nFalse Positives:")
print(false_positives)

print("\nFalse Negatives:")
print(false_negatives)

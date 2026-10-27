"""
Predict whether a user purchases a car using age and estimated salary.
Objectives:
1. Preprocess the dataset and standardise the numerical features.
2.
Train Linear Kernel and RBF Kernel SVM classifiers using the same train-test split.
3. Compare confusion matrices, accuracy, precision, recall, and F1-score and identify
the better-performing kernel.
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Step 1: Load and clean dataset
df = pd.read_csv("Social_Network_Ads_C12.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Step 2: Separate features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Step 4: Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Define the SVM models
models = {
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(kernel="rbf")
}

# Step 6: Train and evaluate
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

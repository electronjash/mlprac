"""
Investigate whether feature scaling affects the prediction of car purchases.
Objectives:
1. Train an RBF Kernel SVM using unscaled age and estimated salary.
2. Train another RBF Kernel SVM using standardised features.
3. Compare their confusion matrices, accuracy, and F1-score and explain the observed differences.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score
)

# Step 1: Load dataset
df = pd.read_csv("Social_Network_Ads_C12.csv")
print(df.head())
print(df.isnull().sum())
df = df.dropna()

# Step 2: Separate inputs and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Step 3: Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Step 4: Train RBF SVM without scaling
model1 = SVC(kernel="rbf")
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

# Step 5: Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train RBF SVM with scaling
model2 = SVC(kernel="rbf")
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)

# Step 7: Compare the results
predictions = {
    "Without Scaling": pred1,
    "With Scaling": pred2
}

accuracy_values = []

for name, pred in predictions.items():

    cm = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    accuracy_values.append(accuracy)

    print("\nModel:", name)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))

# Step 8: Plot results
plt.bar(list(predictions.keys()), accuracy_values)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Effect of Feature Scaling on SVM")
plt.ylim(0, 1)
plt.show()


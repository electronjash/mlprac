
"""
Investigate how the amount of training data influences regression performance.
Objectives:
1. Train regression models using 50%, 70%, and 80% of the dataset for training.
2. Use the same test set wherever applicable, with consistent preprocessing.
3. Compare RMSE and R' Score and discuss whether increasing training data improves
prediction.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("Advertising_clean.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Step 2: Separate features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Reserve 20% for testing
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Shuffle the training data
X_pool = X_pool.sample(frac=1, random_state=42)
y_pool = y_pool.loc[X_pool.index]

# Step 5: Train with different sizes
sizes = [50, 70, 80]
rmse_values = []
r2_values = []

for size in sizes:

    n = int(len(df) * size / 100)

    X_train = X_pool.iloc[:n]
    y_train = y_pool.iloc[:n]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_values.append(rmse)
    r2_values.append(r2)

    print("\nTraining Size:", size, "%")
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# Step 6: Plot the results
plt.plot(sizes, rmse_values, marker="o")

plt.xlabel("Training Data Size (%)")
plt.ylabel("RMSE")
plt.title("Training Size vs RMSE")
plt.grid(True)
plt.show()

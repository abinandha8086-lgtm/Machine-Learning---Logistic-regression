import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)  

print("Predictions:", predictions)
print("Probabilities:\n", probabilities)


try:
    val = float(input("\nEnter a numeric value (e.g., 4.5) to predict class: "))
    sample = np.array([[val]])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    print(f"Input: {val}  => Predicted class: {pred}, Probabilities: {prob}")
except ValueError:
    print("Invalid input. Please enter a number.")

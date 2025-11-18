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

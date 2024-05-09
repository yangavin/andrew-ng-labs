import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression().fit(X, y)

print(f"Actual: {y}")
print(f"Prediction: {model.predict(X)}")
print()
print(f"Accuracy: {model.score(X, y)}")

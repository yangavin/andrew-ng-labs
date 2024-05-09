import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

regressor = LinearRegression().fit(X_train, y_train)
print(f"actual: {y_train}")
print(f"predicted: {regressor.predict(X_train)}")

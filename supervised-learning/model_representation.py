import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2])
y_train = np.array([300, 500])


def compute_model_output(x: np.ndarray, w: float, b: float) -> np.array:
    m = x.shape[0]
    y_hats = np.zeros(m)
    for i in range(m):
        y_hats[i] = w * x[i] + b
    return y_hats


y_pred = compute_model_output(x_train, 200, 100)

plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred)
plt.show()

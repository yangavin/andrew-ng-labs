import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2])
y_train = np.array([300, 500])


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        y_hat = w * x[i] + b
        sum += (y_hat - y[i]) ** 2
    return sum / (2 * m)


w = np.array([])
b = np.array([])
cost = np.array([])
for i in range(300):
    for j in range(300):
        w = np.append(w, i)
        b = np.append(b, j)
        cost = np.append(cost, compute_cost(x_train, y_train, i, j))


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")

ax.plot(w, b, cost)
ax.set
plt.show()

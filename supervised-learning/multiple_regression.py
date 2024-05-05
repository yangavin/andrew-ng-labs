import copy
import numpy as np


def y_hat(x: np.ndarray, w: np.ndarray) -> float:
    return np.dot(x, w)


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = x.shape[0]
    sum = 0
    for i in range(m):
        sum += (y_hat(x[i], w) - y[i]) ** 2
    return sum / (2 * m)


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    m, n = x.shape
    gradient_vector = np.zeros(n)
    for component in range(n):
        sum = 0
        for i in range(m):
            sum += (y_hat(x[i], w) - y[i]) * x[i][component]
        gradient_vector[component] = sum / m
    return gradient_vector


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_start: np.ndarray,
    alpha: float,
    num_iters: int,
):
    cur_w = copy.deepcopy(w_start)
    cost_history = []
    for i in range(num_iters):
        gradient_vector = compute_gradient(x, y, cur_w)
        cur_w -= alpha * gradient_vector
        if i < 100000:
            cost_history.append(compute_cost(x, y, cur_w))
    return cur_w, cost_history


# Driver code
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
X_train = np.array([np.append(feature, 1) for feature in X_train])
w_init = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

w, cost_history = gradient_descent(X_train, y_train, w_init, 5.0e-7, 1000)
print(f"w vector found: {w}")
print()
for i in range(len(X_train)):
    print("actual")
    print(y_train[i])
    print("predicted")
    print(y_hat(X_train[i], w))

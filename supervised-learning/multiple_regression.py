import copy
import numpy as np

np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_train = np.array([np.append(feature, 1) for feature in X_train])

w_init = np.array(
    [0.39133535, 18.75376741, -53.36032453, -26.42131618, 785.1811367994083]
)


def y_hat(x: np.ndarray, w: np.ndarray) -> float:
    return np.dot(x, w)


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = X_train.shape[0]
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
    cost_function: callable,
    gradient_function: callable,
    alpha: float,
    num_iters: int,
):
    cur_w = copy.deepcopy(w_start)
    cost_history = []
    for i in range(num_iters):
        gradient_vector = gradient_function(x, y, w_start)
        cur_w -= alpha * gradient_vector
        if i < 100000:
            cost_history.append(cost_function(x, y, cur_w))
    return cur_w, cost_history


w, cost_history = gradient_descent(
    X_train, y_train, w_init, compute_cost, compute_gradient, 5.0e-7, 1000
)
print(f"w vector found: {w}")

print()

for i in range(len(X_train)):
    print("actual")
    print(y_hat(X_train[i], w))
    print("predicted")
    print(y_train[i])

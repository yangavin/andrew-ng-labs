import numpy as np
import copy


def sigmoid(z: float) -> float | np.ndarray:
    """
    Calculate sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def y_hat(x: np.ndarray, w: np.ndarray):
    """
    Calculate y_hat given an input x and coefficients w
    """
    return sigmoid(np.dot(x, w))


def compute_cost(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, lambda_: float = 0
) -> float:
    """
    Compute cost given a sample set (x, y) and coefficients for the equation (w)
    and regularization constant (lambda_)
    """
    m, n = x.shape
    cost_sum = 0
    for i in range(m):
        cost_sum += -y[i] * np.log(y_hat(x[i], w)) - (1 - y[i]) * np.log(
            1 - y_hat(x[i], w)
        )
    regularization_sum = 0
    for component in range(n):
        regularization_sum += w[component] ** 2
    return cost_sum / m + lambda_ * regularization_sum / (2 * m)


def compute_gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, lamba_: float = 0
) -> np.ndarray:
    """
    Compute the gradient vector for a given cost function (x, y, w, lambda_)
    """
    m, n = x.shape
    gradient_vector = np.zeros(n)
    for component in range(n):
        sum = 0
        for i in range(m):
            sum += (y_hat(x[i], w) - y[i]) * x[i][component]
        gradient_vector[component] = (sum / m) + (lamba_ * w[component] / m)
    return gradient_vector


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    alpha: float,
    iters: int,
    lambda_: float = 0,
) -> tuple[np.ndarray, list]:
    """
    Performs batch gradient descent to learn w. Updates w by taking
    iters gradient steps with learning rate alpha

    Args:
        x :    (array_like Shape (m, n)
        y :    (array_like Shape (m,))
        w_init : (array_like Shape (n,))  Initial values of parameters of the model
        alpha : (float)                 Learning rate
        iters : (int)               number of iterations to run gradient descent
        lambda_ (scalar, float)         regularization constant

    Returns:
        w : (array_like Shape (n,)) Updated values of parameters of the model after
            running gradient descent
        cost_history: (list[float]) History of cost during descent
    """
    cur_w = copy.deepcopy(w_init)
    cost_history = []
    for _ in range(iters):
        cur_w -= alpha * compute_gradient(x, y, cur_w, lambda_)
        cost_history.append(compute_cost(x, y, cur_w, lambda_))
    return cur_w, cost_history


dataset = np.genfromtxt("data/ex2data1.txt", delimiter=",")
x_train = dataset[:, :-1]
x_train = np.array([np.append(example, 1) for example in x_train])
y_train = dataset[:, -1]

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
initial_w = np.append(initial_w, -8)
iterations = 10000
alpha = 0.001
w, cost_history = gradient_descent(x_train, y_train, initial_w, alpha, iterations)

predictions = y_hat(x_train, w)
for i in range(len(predictions)):
    predictions[i] = 1 if predictions[i] > 0.5 else 0
print(f"Accuracy: {np.mean(predictions == y_train)}")

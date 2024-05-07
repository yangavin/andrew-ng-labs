import numpy as np

dataset = np.genfromtxt("data/ex2data1.csv", delimiter=",")
x_train = dataset[:, :-1]
x_train = np.array([np.append(example, 1) for example in x_train])
y_train = dataset[:, -1]


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


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Compute cost given a sample set (x, y) and coefficients for the equation (w)
    """
    m = x.shape[0]
    sum = 0
    for i in range(m):
        sum += -y[i] * np.log(y_hat(x[i], w)) - (1 - y[i]) * np.log(1 - y_hat(x[i], w))
    return sum / m


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the gradient vector for a given cost function (w, y, w)
    """
    m, n = x.shape
    gradient_vector = np.zeros(n)
    for component in range(n):
        sum = 0
        for i in range(m):
            sum += (y_hat(x[i], w) - y[i]) * x[i][component]
        gradient_vector[component] = sum / m
    return gradient_vector

import numpy as np


def linear_regression(x: np.ndarray, y: np.ndarray, rate: float = 0.00001, epochs: int = 10_000):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1

    x = (x - mean) / std

    w = np.zeros(x.shape[1])
    b = 0

    for _ in range(epochs):
        for i in range(len(x)):
            y_predict = np.dot(x[i], w) + b

            w += rate * (y[i] - y_predict) * x[i]
            b += rate * (y[i] - y_predict)

    def predict(x: np.ndarray) -> float:
        return np.dot((x - mean) / std, w) + b

    return predict

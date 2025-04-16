import numpy as np


def linear_regression(x: np.ndarray, y: np.ndarray, rate: float = 0.1, epochs: int = 10_000):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    x = (x - mean) / std

    w = np.zeros(x.shape[1])
    b = 0

    for _ in range(epochs):
        for i in range(len(x)):
            y_predict = np.dot(x[i], w) + b

            for j in range(len(w)):
                w[j] += rate * (y[i] - y_predict) * x[i][j]

            b += rate * (y[i] - y_predict)

    def predict(x) -> float:
        return np.dot((x - mean) / std, w) + b

    return predict

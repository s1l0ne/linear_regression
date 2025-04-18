from typing import Callable

import numpy as np


def MAE(y_true: np.ndarray, y_predict: np.ndarray):
    return np.mean(np.abs(y_true - y_predict))


def MSE(y_true: np.ndarray, y_predict: np.ndarray):
    return np.mean((y_true - y_predict) ** 2)


def r2_score(y_true: np.ndarray, y_predict: np.ndarray):
    return 1 - np.sum((y_true - y_predict) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def evaluate_model(X: np.ndarray, y: np.ndarray, predict_func: Callable[[np.ndarray], float]):
    y_predict = np.array([predict_func(x) for x in X])

    mae = MAE(y, y_predict)
    mse = MSE(y, y_predict)
    rmse = np.sqrt(mse)

    r2 = r2_score(y, y_predict)

    return mae, mse, rmse, r2

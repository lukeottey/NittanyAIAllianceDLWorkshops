import numpy as np

def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)

def nll(y_hat, y):
    return np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))
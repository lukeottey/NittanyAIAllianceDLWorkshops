import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def plot3d(y_hat, y_true, X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], y_hat)
    ax.plot(X[:, 0], X[:, 1], y_true)
    plt.show()


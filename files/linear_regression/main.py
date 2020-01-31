import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, '../../')

from display import plot3d
from loss_functions import mse
from optimization import GradientDescent


class LinearRegression():
    def __init__(self, optimizer):
        self.loss_fn = mse
        self.losses, self.predictions = [], []
        self.model = defaultdict()
        self.optimizer = optimizer

    def predict(self, X):
        if isinstance(X, (int, float, list)):
            X = np.array(X)
        return np.dot(X, self.model['W']) + self.model['b']

    def optimize(self):
        self.optimizer.step()
        for mat, g in self.optimizer.gradient.items():
            self.model[mat] -= g

    def backward(self, X, y, y_hat, m):
        self.optimizer.gradient['W'] = X.T.dot(y_hat - y) / m
        if hasattr(self.optimizer, 'regularizer'):
            self.optimizer.regularize(mat='W')
        self.optimizer.gradient['b'] = (y_hat - y).mean()

    def initialize(self, n_features):
        self.model['W'] = np.zeros(n_features)
        self.model['b'] = 0.
        self.optimizer.gradient['W'] = np.zeros(n_features)
        self.optimizer.gradient['b'] = 0.

    def fit(self, X, y, epochs):
        m_examples, n_features = X.shape
        self.initialize(n_features)
        for e in range(epochs):
            y_hat = self.predict(X)
            self.predictions.append(y_hat)
            self.losses.append(self.loss_fn(y_hat, y))
            self.backward(X, y, y_hat, m_examples)
            self.optimize()


def main():
    x1, x2 = np.arange(10, step=0.025), np.arange(10, step=0.025)
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    X = np.column_stack((x1, x2))
    # y = np.array([np.sin(x0) + np.cos(x1) for x0, x1 in X])
    y = np.array([np.sin(x0) * x1 for x0, x1 in X])
    optimizer = GradientDescent(lr=0.01, regularizer=None)
    model = LinearRegression(optimizer)

    model.fit(X, y, epochs=5000)
    for i in range(0, 5000, 500):
        print('epoch [{}]: {}'.format(i, round(model.losses[i], 5)))

    y_hat = model.predictions[-1]
    
    plot3d(y_hat, y, X)

if __name__ == '__main__':
    main()

import numpy as np
from collections import defaultdict

from loss_functions import mse

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
            self.losses.append(np.mean(self.loss_fn(y_hat, y)))
            self.backward(X, y, y_hat, m_examples)
            self.optimize()

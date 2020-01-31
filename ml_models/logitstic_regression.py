import numpy as np
from ml_models import LinearRegression

__all__ = ['LogisticRegression']

class LogisticRegression(LinearRegression):
    def __init__(self, optimizer, loss):
        super(LogisticRegression, self).__init__(optimizer)
        self.loss_fn = loss

    def logistic_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self.logistic_sigmoid(np.dot(X, self.model['W']) + self.model['b'])

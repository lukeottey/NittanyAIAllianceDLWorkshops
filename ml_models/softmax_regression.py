import numpy as np
from ml_models import LinearRegression

__all__ = ['SoftmaxRegression']

class SoftmaxRegression(LinearRegression):
    def __init__(self, optimizer):
        super(SoftmaxRegression, self).__init__(optimizer)
        self.loss_fn = lambda y_hat, y: np.mean(-np.sum(y * np.log(y_hat), axis=1))

    def predict(self, X):
        output = np.dot(X, self.model['W']) + self.model['b']
        classes = np.arange(self.n_classes)
        x = output - np.max(output)
        pred = classes[np.argmax(np.exp(x)/sum(np.exp(x)), axis=1)]
        pred = np.eye(self.n_classes)[pred].astype(float)
        return pred

    def initialize(self, n_classes, n_features):
        self.model['W'] = np.zeros((n_features, n_classes))
        self.model['b'] = np.zeros((1, n_classes))
        self.optimizer.gradient['W'] = np.zeros_like(self.model['W'])
        self.optimizer.gradient['b'] = np.zeros_like(self.model['b'])
        self.n_classes = n_classes

    def fit(self, X, y, epochs):
        c = y.shape[1]
        m_examples, n_features = X.shape
        self.initialize(n_classes=c, n_features=n_features)
        for e in range(epochs):
            y_hat = self.predict(X)
            self.loss.append(self.loss_fn(y_hat, y))
            print(self.loss[-1])
            self.backward(X, y, y_hat, m_examples)
            self.optimize()



import numpy as np
from collections import defaultdict

__all__ = ['LinearRegression', 'LogisticRegression', 'SoftmaxRegression']

class LinearRegression:
    def __init__(self, optimizer, loss):
        self.loss_fn = loss
        self.logger = defaultdict(list)
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

    def adjust_lr(self, gamma):
        self.optimizer.lr *= gamma

    def backward(self, X, y, y_hat, m):
        self.optimizer.gradient['W'] = X.T.dot(y_hat - y) / m
        if hasattr(self.optimizer, 'regularizer'):
            self.optimizer.regularize(mat='W')
        self.optimizer.gradient['b'] = (y_hat - y).mean()

    def initialize(self, num_features):
        self.model['W'] = np.random.normal(loc=0.0, scale=0.01, size=(num_features, ))
        self.model['b'] = 0.
        self.optimizer.gradient['W'] = np.zeros(num_features)
        self.optimizer.gradient['b'] = 0.
    
    def fit(self, X, y):
        y_hat = self.predict(X)
        loss = np.mean(self.loss_fn(y_hat, y))
        self.backward(X, y, y_hat, X.shape[0])
        self.optimize()
        self.logger['train_loss'].append(loss)
        return loss

    def evaluate(self, X, y):
        y_hat = self.predict(X)
        loss = np.mean(self.loss_fn(y_hat, y))
        self.logger['test_loss'].append(loss)
        return loss

class LogisticRegression(LinearRegression):
    def __init__(self, optimizer, loss):
        super(LogisticRegression, self).__init__(optimizer, loss)

    def logistic_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self.logistic_sigmoid(np.dot(X, self.model['W']) + self.model['b'])

    def fit(self, X, y):
        y_hat = self.predict(X)
        accuracy = np.sum(((y_hat > 0.5).astype('int') == y).astype('int')) / len(y_hat)
        loss = np.mean(self.loss_fn(y_hat, y))
        self.backward(X, y, y_hat, X.shape[0])
        self.optimize()
        self.logger['train_accuracy'].append(accuracy)
        self.logger['train_loss'].append(loss)
        return accuracy, loss

    def evaluate(self, X, y):
        y_hat = self.predict(X)
        accuracy = np.sum(((y_hat > 0.5).astype('int') == y).astype('int')) / len(y_hat)
        loss = np.mean(self.loss_fn(y_hat, y))
        self.logger['test_accuracy'].append(accuracy)
        self.logger['test_loss'].append(loss)
        return accuracy, loss

class SoftmaxRegression(LinearRegression):
    epsilon = 1e-6
    def __init__(self, optimizer, loss, num_classes=10):
        super(SoftmaxRegression, self).__init__(optimizer, loss)
        self.num_classes = num_classes
    
    def predict(self, X):
        output = np.dot(X, self.model['W']) + self.model['b']
        classes = np.arange(self.num_classes)
        x = output - np.max(output)
        y_hat = classes[np.argmax(np.exp(x)/sum(np.exp(x)), axis=1)]
        y_hat = np.eye(self.num_classes)[y_hat].astype(float) + self.epsilon
        return y_hat

    def initialize(self, num_features):
        self.model['W'] = np.random.normal(loc=0.0, scale=0.01, size=(num_features, self.num_classes))
        self.model['b'] = np.zeros((1, self.num_classes))
        self.optimizer.gradient['W'] = np.zeros_like(self.model['W'])
        self.optimizer.gradient['b'] = np.zeros_like(self.model['b'])

    def fit(self, X, y):
        y_hat = np.clip(self.predict(X), self.epsilon, 1.0 - self.epsilon)
        y = np.eye(self.num_classes)[y].astype(float)
        y = np.clip(y, self.epsilon, 1.0 - self.epsilon)
        loss = self.loss_fn(y_hat, y)
        accuracy = np.sum((np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)).astype('int')) / len(y)
        self.backward(X, y, y_hat, X.shape[0])
        self.optimize()
        self.logger['train_acc'].append(accuracy)
        self.logger['train_loss'].append(loss)
        return accuracy, loss

    def evaluate(self, X, y):
        y_hat = np.clip(self.predict(X), self.epsilon, 1.0 - self.epsilon)
        y = np.eye(self.num_classes)[y].astype(float)
        y = np.clip(y, self.epsilon, 1.0 - self.epsilon)
        loss = np.mean(self.loss_fn(y_hat, y))
        accuracy = np.sum((np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)).astype('int')) / len(y)
        self.logger['test_acc'].append(accuracy)
        self.logger['test_loss'].append(loss)
        return accuracy, loss
    



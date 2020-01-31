import sys
import numpy as np
from scipy.optimize import rosen as rosenbrock_fn

sys.path.insert(0, '../../')
from display import plot3d
from files import LinearRegression
from loss_functions import mse, nll
from optimization import Momentum

__all__ = ['LogisticRegression']


class LogisticRegression(LinearRegression):
    def __init__(self, optimizer, loss):
        super(LogisticRegression, self).__init__(optimizer)
        self.loss_fn = loss

    def logistic_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self.logistic_sigmoid(np.dot(X, self.model['W']) + self.model['b'])



def main():
    epochs = 50000
    x1, x2 = np.arange(10, step=0.0025) - 5, np.arange(10, step=0.0025) - 5
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    X = np.column_stack((x1, x2))
    y = np.array([rosenbrock_fn(x) for x in X])
    y_true = y #(y > y_avg).astype('float')
    
    optimizer = Momentum(lr=0.1, momentum=0.9, regularizer=None)
    model = LogisticRegression(optimizer)
    model.fit(X, y_true, epochs=epochs)

    for i in range(0, epochs, 1000):
        print('epoch [{}]: {}'.format(i, round(model.losses[i], 5)))

    y_hat = model.predictions[-1].astype('float')


    correct = (y_true == y_hat).astype('float')
    accuracy = np.sum(correct) / len(correct)
    print(accuracy)

    plot3d(y_hat, y_true, X)

if __name__ == '__main__':
    main()

import numpy as np

__all__ = ['Regularizer', 'L1Regularizer', 'L2Regularizer', 'ElasticRegularizer']

class Regularizer:
    def __init__(self, lambd):
        self._lambd = lambd

    def __call__(self, W):
        raise NotImplementedError()

class L1Regularizer(Regularizer):
    def __init__(self, lambd):
        super(L1Regularizer, self).__init__(lambd)

    def __call__(self, W):
        return self._lambd * np.where(W >= 0, 1, -1)

class L2Regularizer(Regularizer):
    def __init__(self, lambd):
        super(L2Regularizer, self).__init__(lambd)

    def __call__(self, W):
        return self._lambd * W

class ElasticRegularizer(Regularizer):
    def __init__(self, lambd, ratio):
        self.__l1 = L1Regularizer(lambd)
        self.__l2 = L2Regularizer(lambd)
        self.__ratio = ratio

    def __call__(self, W):
        return self.__ratio * self.__l1.__call__(W) + (1 - self.__ratio) * self.__l2.__call__(W)

from collections import defaultdict

class Optimizer:
    def __init__(self, lr, regularizer=None):
        self.lr = lr
        self.gradient = defaultdict()
        if regularizer is not None:
            self.regularizer = regularizer
    
    def regularize(self, mat):
        self.gradient[mat] += self.regularizer(self.gradient[mat])

    def step(self):
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, lr, regularizer=None):
        super(GradientDescent, self).__init__(lr, regularizer)

    def step(self):
        for mat, g in self.gradient.items():
            self.gradient[mat] = self.lr * g

class Momentum(Optimizer):
    def __init__(self, lr, momentum, regularizer=None):
        super(Momentum, self).__init__(lr, regularizer=regularizer)
        self.momentum = momentum

    def step(self):
        n = len(self.gradient.keys())
        if not hasattr(self, 'velocity'):
            self.velocity = defaultdict()
            for mat in self.gradient.keys():
                self.velocity[mat] = 0
        for mat, g in self.gradient.items():
            self.velocity[mat] = (1 - self.momentum) * g + self.momentum * self.velocity[mat]
            self.gradient[mat] = self.lr * self.velocity[mat]
            
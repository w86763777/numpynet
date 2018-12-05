import numpy as np


class L1():
    def __init__(self, weight):
        self.weight = weight

    def loss(self, W):
        return self.weight * np.sum(np.abs(W))

    def grad(self, W):
        return self.weight * np.sign(W)

    def __str__(self):
        return "L1(weight=%f)" % self.weight


class L2():
    def __init__(self, weight):
        self.weight = weight

    def loss(self, W):
        return np.sum(self.weight * W * W) / 2

    def grad(self, W):
        return self.weight * W

    def __str__(self):
        return "L2(weight=%f)" % self.weight

import numpy as np
import abc

from numpynet.constants import eps
from numpynet.utils import assertfinite
from numpynet.metrics import (
    binary_cross_entropy, mean_squared_error, cross_entropy)


class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, x, y):
        return NotImplemented

    @abc.abstractmethod
    def backward(self):
        return NotImplemented

    @abc.abstractmethod
    def get_metric(self):
        return NotImplemented


class BinaryCrossEntropy(Loss):
    def forward(self, x, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=-1)
        self.old_x = x
        self.old_y = y
        return -(y * np.log(x + eps) + (1 - y) * np.log(1 - x + eps))

    def backward(self):
        grad = (
            (1 - self.old_y) / (1 - self.old_x + eps) -
            self.old_y / (self.old_x + eps))
        assertfinite(grad)
        return grad

    def get_metric(self):
        return binary_cross_entropy


class MeanSquaredError(Loss):
    def forward(self, x, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=-1)
        self.old_x = x
        self.old_y = y
        return np.mean((x - y) * (x - y))

    def backward(self):
        return 2 * (self.old_x - self.old_y)

    def get_metric(self):
        return mean_squared_error


class CrossEntropy(Loss):

    def forward(self, x, y):
        self.old_x = x.clip(min=eps, max=None)
        self.old_y = y
        loss = np.sum(np.where(self.old_y == 1,
                               -np.log(self.old_x), 0), axis=-1)
        loss = np.mean(loss)
        return loss

    def backward(self):
        return np.where(self.old_y == 1, -1 / self.old_x, 0)

    def get_metric(self):
        return cross_entropy

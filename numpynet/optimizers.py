import abc
import numpy as np

from numpynet.constants import eps
from numpynet.utils import exp_running_avg, assertfinite


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(self, model):
        return NotImplemented


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimize(self, model):
        for layer in model.layers:
            for para, grad in layer.parameters():
                para -= self.learning_rate * grad


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, gamma=0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.cache = {}

    def optimize(self, model):
        for layer in model.layers:
            parameters = layer.parameters()
            cache = self.cache.get(layer, [None for _ in range(len(parameters))])
            for i, (para, grad) in enumerate(parameters):
                cache[i] = exp_running_avg(cache[i], grad**2, self.gamma)
                para -= self.learning_rate * grad / (np.sqrt(cache[i]) + eps)
                assertfinite(para)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_exp = 1
        self.beta2_exp = 1
        self.caches_m = {}
        self.caches_v = {}

    def optimize(self, model):
        self.beta1_exp *= self.beta1
        self.beta2_exp *= self.beta2
        for layer in model.layers:
            parameters = layer.parameters()
            cache_m = self.caches_m.get(layer, [None] * len(parameters))
            cache_v = self.caches_v.get(layer, [None] * len(parameters))
            for i, (para, grad) in enumerate(parameters):
                cache_m[i] = exp_running_avg(cache_m[i], grad, self.beta1)
                cache_v[i] = exp_running_avg(cache_v[i], grad ** 2, self.beta2)
                m_head = cache_m[i] / (1 - self.beta1_exp)
                v_head = cache_v[i] / (1 - self.beta2_exp)
                para -= self.learning_rate * m_head / (np.sqrt(v_head) + eps)
                assertfinite(para)

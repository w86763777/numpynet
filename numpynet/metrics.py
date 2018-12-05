import numpy as np

from numpynet.constants import eps


def root_mean_squared_error(y_true, y_pred):
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, axis=-1)
    return np.mean(np.sqrt(np.mean(np.square(y_true - y_pred), axis=-1)))


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def categorical_accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    return (y_true == y_pred).sum() / len(y_true)


def mean_squared_error(y_true, y_pred):
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, axis=-1)
    return np.mean((y_pred - y_true) * (y_pred - y_true))


def cross_entropy(y_true, y_pred):
    y_pred = y_pred.clip(min=eps, max=None)
    loss = np.sum(np.where(y_true == 1, -np.log(y_pred), 0), axis=-1)
    loss = np.mean(loss)
    return loss


def binary_cross_entropy(y_true, y_pred):
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, axis=-1)
    return (
        - (y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)))

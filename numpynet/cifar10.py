import pickle
import numpy as np
import os

from munch import munchify
from numpynet.utils import onehot


def read_data_sets(path='./dataset/cifar-10', one_hot=True):
    X_train = []
    y_train = []
    for k in range(5):
        X, y = load_data_batch(path, k + 1)
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train, axis=0)
    y_train = onehot(np.concatenate(y_train, axis=0), 10)
    X_valid = X_train[:5000]
    y_valid = y_train[:5000]
    X_train = X_train[5000:]
    y_train = y_train[5000:]
    X_test, y_test = load_test_batch(path)
    y_test = onehot(y_test, 10)

    return munchify({
        'train': {
            'images': X_train,
            'labels': y_train,
        },
        'validation': {
            'images': X_valid,
            'labels': y_valid,
        },
        'test': {
            'images': X_test,
            'labels': y_test,
        }
    })


def label_names(path='./dataset/cifar-10'):
    return pickle.load(open(os.path.join(path, 'batches.meta'), 'rb'))


def load_data_batch(path, k):
    data_batch_name = 'data_batch_%d' % k
    return _load_batch(os.path.join(path, data_batch_name))


def load_test_batch(path):
    data_batch_name = 'test_batch'
    return _load_batch(os.path.join(path, data_batch_name))


def _load_batch(path):
    with open(path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    X = batch[b'data'] / 255.
    y = np.array(batch[b'labels'])
    return X, y

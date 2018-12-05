import csv
import numpy as np
import os

from munch import munchify
from numpynet.utils import onehot
from numpynet.dataset import _get_data


_ids = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}
_label_names = ['setosa', 'versicolor', 'virginica']


def read_data_sets(one_hot=True):
    with open(_get_data('data/iris.csv'), newline='') as csvfile:
        rows = csv.reader(csvfile)
        X = np.zeros((150, 4))
        y = np.zeros(150, dtype=np.int32)
        for i, row in enumerate(rows):
            X[i] = np.array(row[:4])
            y[i] = _ids[row[4]]
        if one_hot:
            y = onehot(y, 3)
        return X, y


def label_names():
    return _label_names
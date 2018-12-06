import csv
import os
import gzip
import urllib.request
import numpy as np
from munch import Munch
from tqdm import tqdm

from numpynet.dataset import Dataset
from numpynet.utils import onehot


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, chunks=1, chunk_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(chunks * chunk_size - self.n)
        

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def read_data_sets(dir_path, one_hot=True, normalize=True):
    base_url = 'http://yann.lecun.com/exdb/mnist/%s'
    gzs = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]
    os.makedirs(dir_path, exist_ok=True)
    for gz in gzs:
        url = base_url % gz
        path = os.path.join(dir_path, gz)
        if not os.path.exists(path):
            # all optional kwargs
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=gz) as t:
                urllib.request.urlretrieve(url, path, t.update_to)

    X_train = extract_images(os.path.join(dir_path, gzs[0]))
    y_train = extract_labels(os.path.join(dir_path, gzs[1]))
    X_train, X_val = X_train[5000:], X_train[:5000]
    y_train, y_val = y_train[5000:], y_train[:5000]
    X_test = extract_images(os.path.join(dir_path, gzs[2]))
    y_test = extract_labels(os.path.join(dir_path, gzs[3]))

    if one_hot:
        y_train = onehot(y_train, 10)
        y_val = onehot(y_val, 10)
        y_test = onehot(y_test, 10)
    
    if normalize:
        X_train = X_train / 255.
        X_val = X_val / 255.
        X_test = X_test / 255.
    
    return Munch(
        train=Dataset(X=X_train, y=y_train),
        val=Dataset(X=X_val, y=y_val),
        test=Dataset(X=X_test, y=y_test))

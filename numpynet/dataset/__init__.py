import numpy as np
import os


def _get_data(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)


def split_data(x, *args, **kwargs):
    test_size = kwargs.get('test_size', 0.33)
    shuffle = kwargs.get('shuffle', True)

    if shuffle:
        index = np.random.permutation(len(x))
    else:
        index = np.arange(len(x))
    split_index = int(len(x) * (1 - test_size))
    train_index, test_index = index[:split_index], index[split_index:]
    if len(args) == 0:
        return x[train_index], x[test_index]
    else:
        ret = [x[train_index], x[test_index]]
        for arg in args:
            ret.append(arg[train_index])
            ret.append(arg[test_index])
        return tuple(ret)


def split_dataset(dataset, **kwargs):
    test_size = kwargs.get('test_size', 0.33)
    shuffle = kwargs.get('shuffle', True)

    if shuffle:
        index = np.random.permutation(len(dataset.X))
    else:
        index = np.arange(len(dataset.X))
    split_index = int(len(dataset.X) * (1 - test_size))
    train_index, test_index = index[:split_index], index[split_index:]
    
    return (
        Dataset(X=dataset.X[train_index], y=dataset.y[train_index]),
        Dataset(X=dataset.X[test_index], y=dataset.y[test_index]),
    )

def generator(data, batch_size, shuffle=True):
    if type(data) is list or type(data) is tuple:
        length = data[0].shape[0]
    else:
        length = len(data)

    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    start_index = 0
    while start_index < length:
        if type(data) is list or type(data) is tuple:
            batch = []
            for x in data:
                batch.append(x[indices[start_index: start_index + batch_size]])
            batch = tuple(batch)
        else:
            batch = data[indices[start_index: start_index + batch_size]]
        start_index += batch_size
        yield batch


class Dataset:
    def __init__(self, X, y, batch_size=128, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert(X.shape[0] == y.shape[0])

    def __iter__(self):
        length = self.y.shape[0]
        indices = np.arange(length)
        if self.shuffle:
            np.random.shuffle(indices)
        start = 0
        while start < length:
            batch_X = self.X[indices[start: start + self.batch_size]]
            batch_y = self.y[indices[start: start + self.batch_size]]
            start += self.batch_size
            yield batch_X, batch_y

    def __len__(self):
        length = self.y.shape[0]
        num_batch = length // self.batch_size + (length % self.batch_size != 0)
        return num_batch

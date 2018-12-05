import pickle

from nctu.nn.cifar10 import read_data_sets
from nctu.nn.dataset import Dataset
from nctu.nn.metrics import categorical_accuracy
from nctu.nn.models import SequentialModel
from nctu.nn.regularizers import L2
from nctu.nn.optimizers import RMSprop
from nctu.nn.loss import CrossEntropy
from nctu.nn.callbacks import ExtraValidation, Checkpoint
from nctu.nn.layers import (
    Dense, Convolution2D, Flatten, ReLU, Softmax, Dropout, Maxpooling2D, Input)

learning_rate = 0.001
batch_size = 200
epochs = 40
reg_w = 0.0001

if __name__ == "__main__":
    cifar10 = read_data_sets('./dataset/cifar10', one_hot=True)
    train = Dataset(
        X=cifar10.train.images.reshape(-1, 3, 32, 32),
        y=cifar10.train.labels, batch_size=batch_size)
    val = Dataset(
        X=cifar10.validation.images.reshape(-1, 3, 32, 32),
        y=cifar10.validation.labels, batch_size=batch_size)
    test = Dataset(
        X=cifar10.test.images.reshape(-1, 3, 32, 32),
        y=cifar10.test.labels, batch_size=batch_size)

    model = SequentialModel()
    model.add(Input((3, 32, 32)))

    model.add(Convolution2D(
        16, size=[3, 3], stride=[1, 1], regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    # model.add(Dropout(0.2))

    model.add(Convolution2D(
        32, size=[3, 3], stride=[1, 1], regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(
        128, kernel_regularizer=L2(reg_w), bias_regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(
        10, kernel_regularizer=L2(reg_w), bias_regularizer=L2(reg_w)))
    model.add(Softmax())

    model.compile(
        objective=CrossEntropy(),
        optimizer=RMSprop(learning_rate=learning_rate),
        metric=[categorical_accuracy])

    print(model)
    history = ExtraValidation(test)
    model.fit(
        x=train.X, y=train.y, val_x=val.X, val_y=val.y, epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            history,
            Checkpoint(
                './models/cifar10/nn3x3s1.r.model.pkl', categorical_accuracy),
        ])

    with open('./models/cifar10/nn3x3s1.r.history.pkl', 'wb') as handle:
        pickle.dump(history.history, handle)

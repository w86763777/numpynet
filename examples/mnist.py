import pickle
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

from numpynet.dataset import Dataset
from numpynet.metrics import categorical_accuracy
from numpynet.regularizers import L2
from numpynet.models import SequentialModel
from numpynet.optimizers import RMSprop
from numpynet.loss import CrossEntropy
from numpynet.callbacks import ExtraValidation, Checkpoint
from numpynet.layers import (
    Input, Dense, Convolution2D, Flatten, ReLU, Softmax, Dropout, Maxpooling2D)

learning_rate = 0.001
batch_size = 200
epochs = 20
reg_w = 0.00001


if __name__ == "__main__":
    # data preprocessing
    mnist = read_data_sets("./dataset/mnist", one_hot=True)
    train = Dataset(
        X=mnist.train.images.reshape(-1, 1, 28, 28),
        y=mnist.train.labels, batch_size=batch_size)
    val = Dataset(
        X=mnist.validation.images.reshape(-1, 1, 28, 28),
        y=mnist.validation.labels, batch_size=batch_size)
    test = Dataset(
        X=mnist.test.images.reshape(-1, 1, 28, 28),
        y=mnist.test.labels, batch_size=batch_size)

    model = SequentialModel()
    model.add(Input((1, 28, 28)))
    model.add(Convolution2D(
        8, size=[3, 3], stride=[1, 1], regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    model.add(Convolution2D(
        16, size=[3, 3], stride=[1, 1], regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(
        128, kernel_regularizer=L2(reg_w)))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(
        10, kernel_regularizer=L2(reg_w)))
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
                './models/mnist/nn3x3s1.r,model.pkl', categorical_accuracy),
        ])

    with open('./models/mnist/nn3x3s1.r.history.pkl', 'wb') as handle:
        pickle.dump(history.history, handle)

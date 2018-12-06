import pickle

from numpynet.dataset import mnist
from numpynet.metrics import categorical_accuracy
from numpynet.models import SequentialModel
from numpynet.optimizers import RMSprop
from numpynet.loss import CrossEntropy
from numpynet.callbacks import ExtraValidation, Checkpoint
from numpynet.layers import (
    Input, Dense, Convolution2D, Flatten, ReLU, Softmax, Dropout, Maxpooling2D)


if __name__ == "__main__":
    mnist = mnist.read_data_sets('./mnist', one_hot=True)

    model = SequentialModel()
    model.add(Input((1, 28, 28)))
    model.add(Convolution2D(8, size=[3, 3], stride=[1, 1]))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    model.add(Convolution2D(16, size=[3, 3], stride=[1, 1]))
    model.add(ReLU())
    model.add(Maxpooling2D(size=[3, 3], stride=[2, 2]))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Softmax())

    model.compile(
        objective=CrossEntropy(),
        optimizer=RMSprop(learning_rate=0.001),
        metric=[categorical_accuracy])

    print(model)
    history = ExtraValidation(mnist.test)
    model.fit(
        x=mnist.train.X, y=mnist.train.y, val_x=mnist.val.X, val_y=mnist.val.y,
        epochs=1, batch_size=200,
        callbacks=[
            history,
            Checkpoint('./best_model.pkl', categorical_accuracy),
        ])

    with open('./history.pkl', 'wb') as handle:
        pickle.dump(history.history, handle)

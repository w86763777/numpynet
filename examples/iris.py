from numpynet.dataset import iris, split_dataset
from numpynet.models import SequentialModel
from numpynet.optimizers import Adam
from numpynet.loss import CrossEntropy
from numpynet.metrics import categorical_accuracy
from numpynet.layers import Input, Dense, ReLU, Softmax, Dropout


if __name__ == "__main__":
    # load iris dataset
    iris = iris.read_data_sets()
    # split dataset
    train, test = split_dataset(iris, test_size=0.33)

    # build model
    model = SequentialModel()
    model.add(Input((4,)))
    model.add(Dense(10))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Softmax())

    # assign objective, optimizer and metrics which is going to be shown on
    # progress bar
    model.compile(
        objective=CrossEntropy(),
        optimizer=Adam(learning_rate=0.001),
        metric=[categorical_accuracy])
    
    # fit on data
    model.fit(
        x=train.X, y=train.y, val_x=test.X, val_y=test.y,
        epochs=500, batch_size=8)

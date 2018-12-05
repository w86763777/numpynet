# NumpyNet

High level neural network API implementated using numpy

# Requirements
- python3

# Install
```
$ git clone https://github.com/w86763777/numpynet
$ cd numpynet
$ python setup.py install
```

# Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from numpynet.dataset import Dataset
from numpynet.models import SequentialModel
from numpynet.optimizers import RMSprop
from numpynet.loss import CrossEntropy
from numpynet.utils import onehot
from numpynet.metrics import categorical_accuracy
from numpynet.layers import Input, Dense, ReLU, Softmax, Dropout

learning_rate = 0.001
dropout_rate = 0.5
epochs = 1000
batch_size = 32

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_informative=3, n_features=50, n_classes=4)
    y = onehot(y, 4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train = Dataset(X_train, y_train, batch_size)
    test = Dataset(X_test, y_test, batch_size)

    model = SequentialModel()
    model.add(Input((50,)))
    model.add(Dense(30))
    model.add(ReLU())
    model.add(Dropout(dropout_rate))
    model.add(Dense(15))
    model.add(ReLU())
    model.add(Dropout(dropout_rate))
    model.add(Dense(4))
    model.add(Softmax())
    
    model.compile(
        objective=CrossEntropy(),
        optimizer=RMSprop(learning_rate=learning_rate),
        metric=[categorical_accuracy])
    
    model.fit(
        x=train.X, y=train.y, val_x=test.X, val_y=test.y, epochs=epochs,
        batch_size=batch_size)

```
- output
```
Epoch 999/1000
100%|█████████████| 24/24 [00:00<00:00, 731.42it/s, categorical_accuracy=0.7987, cross_entropy=0.7114, val_categorical_accuracy=0.6960, val_cross_entropy=1.9230]
Epoch 1000/1000
100%|█████████████| 24/24 [00:00<00:00, 691.88it/s, categorical_accuracy=0.8133, cross_entropy=0.5209, val_categorical_accuracy=0.6920, val_cross_entropy=1.9149]
```
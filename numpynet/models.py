import numpy as np
import pickle
import os
from tqdm import tqdm

from numpynet.utils import assertfinite
from numpynet.dataset import generator, Dataset
from numpynet.layers import Input
from numpynet.callbacks import History


class SequentialModel():
    def __init__(self, layers=[]):
        """Construct model with specific layers"""
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        """Add a layer to the end of sequential model."""
        if len(self.layers) == 0:
            assert(type(layer) == Input)
        else:
            layer = layer(self.layers[-1])
        self.layers.append(layer)

    def compile(self, objective, optimizer, metric=[]):
        """Set the objective function, optimizer and metrics."""
        self.objective = objective
        self.optimizer = optimizer
        self.metrics = [objective.get_metric()] + metric

    def train(self, x_batch, y_batch):
        """Train the model on batch of data."""
        y_pred = self._forward(x_batch)
        self._loss(y_batch, y_pred)
        self._backward()
        return self._evaluate_metrics(y_batch, y_pred)

    def predict(self, X, batch_size=128):
        """Predict the input and return the output of last layer."""
        self._set_testing()
        y_pred = []
        for x in generator(X, batch_size, shuffle=False):
            y_ = self._forward(x)
            y_pred.append(y_)
        y_pred = np.concatenate(y_pred, axis=0)
        self._set_training()
        return y_pred

    def evaluate(self, X, y, batch_size=128):
        """Predict and evaluate loss and metrics

        Return:
            if len(metrics) == 0 then return single numerical loss, otherwise
            return a list containing loss at first element and all the metric
            results
        """
        y_pred = self.predict(X, batch_size=batch_size)
        return self._evaluate_metrics(y, y_pred)

    def _evaluate_metrics(self, y_true, y_pred):
        """Evaluate loss and metrics.

        Return:
            if len(metrics) == 0 then return single numerical loss, otherwise
            return a list containing loss at first element and all the metric
            results

        """
        evaluation = [metric(y_true, y_pred) for metric in self.metrics]
        return np.array(evaluation)

    def _forward(self, x):
        """Forward propogation through layers."""
        assertfinite(x)
        for layer in self.layers:
            x = layer.forward(x)
            assertfinite(x)
        return x

    def _backward(self):
        """Backward propogation through layers."""
        grad = self.objective.backward()
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backward(grad)
        self.optimizer.optimize(self)

    def _loss(self, y_true, y_pred):
        """Compute loss with regularization penalty.

        The function makes objective function cache inputs for computing
        gradients.

        """
        regularization_loss = 0
        for layer in self.layers:
            regularization_loss += layer.regularization_loss()
        return self.objective.forward(y_pred, y_true) + regularization_loss

    def _set_testing(self):
        """Set layers in testing mode to prevent caching inputs."""
        for layer in self.layers:
            layer.set_testing(is_testing=True)

    def _set_training(self):
        """Set layers in training mode to enforce caching inputs."""
        for layer in self.layers:
            layer.set_testing(is_testing=False)

    def save(self, path):
        """Save model in pickle formate"""
        if len(os.path.dirname(path)) != 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def fit(self, x, y, val_x=None, val_y=None, epochs=1, batch_size=128,
            split=0.2, callbacks=[]):
        if val_x is None or val_y is None:
            val_x, x = x[:len(x) * split], x[len(x) * split:]
            y, val_y = y[:len(y) * split], y[len(y) * split:]
        train = Dataset(X=x, y=y, batch_size=batch_size)

        history = History()
        callbacks.append(history)
        # Fire train callbacks
        for callback in callbacks:
            callback.on_train_begin(self)
        # Start training
        for epoch in range(epochs):
            # Fire epoch callbacks
            for callback in callbacks:
                callback.on_epoch_begin(self, epoch + 1)
            print('Epoch %d/%d' % (epoch + 1, epochs))
            with tqdm(total=len(train)) as pbar:
                rolling_sums = np.zeros(len(self.metrics))
                counter = 0
                for batch, (x_batch, y_batch) in enumerate(train):
                    # Fire batch callbacks
                    for callback in callbacks:
                        callback.on_batch_begin(self, epoch + 1, batch + 1)
                    counter += len(x_batch)
                    evals = self.train(x_batch, y_batch)
                    rolling_sums += evals * len(x_batch)
                    batch_log = {}
                    msg = {}
                    for s, e, metric in zip(rolling_sums, evals, self.metrics):
                        batch_log[metric.__name__] = e
                        msg[metric.__name__] = '%.4f' % (s / counter)
                    pbar.set_postfix(**msg)
                    pbar.update(1)
                    # Fire batch callbacks
                    for callback in callbacks:
                        callback.on_batch_end(
                            self, epoch + 1, batch + 1, batch_log)
                evals = self.evaluate(val_x, val_y)
                epoch_log = {}
                for e, metric in zip(evals, self.metrics):
                    epoch_log['val_' + metric.__name__] = e
                    msg['val_' + metric.__name__] = '%.4f' % e
                pbar.set_postfix(**msg)
            # Fire epoch callbacks
            for callback in callbacks:
                callback.on_epoch_end(self, epoch + 1, epoch_log)
        # Fire train callbacks
        for callback in callbacks:
            callback.on_train_end(self)
        return history

    def __str__(self):
        return (
            "[\n\t" + ",\n\t".join(str(layer) for layer in self.layers) + "\n]"
        )

import numpy as np
import abc

from numpynet.constants import eps
from numpynet.utils import (
    exp_running_avg, assertfinite, im2col_indices, col2im_indices)


class Layer(abc.ABC):
    def __init__(self):
        self.is_testing = False

    @abc.abstractmethod
    def forward(self):
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad):
        return NotImplemented

    @abc.abstractmethod
    def compute_output_shape(self):
        return NotImplemented

    def regularization_loss(self):
        return 0

    def set_testing(self, is_testing):
        self.is_testing = is_testing

    def parameters(self):
        return []


class Input(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, x):
        assert(len(x.shape) == len(self.input_shape) + 1)
        return x

    def backward(self, grad):
        return np.zeros_like(grad)

    def compute_output_shape(self):
        return self.input_shape

    def __str__(self):
        return "Input(input_shape=%s)" % str(self.input_shape)


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        if dropout_rate > 1 or dropout_rate < 0:
            raise ValueError('dropout_rate should be in range [0., 1.]')
        self.dropout_keep_rate = 1 - dropout_rate

    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        return self

    def forward(self, x):
        if self.is_testing:
            return x
        else:
            self.old_mask = np.random.binomial(
                1, self.dropout_keep_rate, x.shape[1:])
            return x * self.old_mask / self.dropout_keep_rate

    def backward(self, grad):
        if self.is_testing:
            return grad
        else:
            return grad * self.old_mask / self.dropout_keep_rate

    def compute_output_shape(self):
        return self.output_shape

    def __str__(self):
        return "Dropout(drouput_rate=%f)" % (1 - self.dropout_keep_rate)


class Linear(Layer):
    def __init__(self, output_units, regularizer=None):
        super().__init__()
        self.output_units = output_units
        self.regularizer = regularizer

    def __call__(self, pre_layer):
        pre_shape = pre_layer.compute_output_shape()
        assert(len(pre_shape) == 1)
        self.input_units = pre_shape[0]
        self.W = np.random.normal(
            loc=0, scale=0.1,
            size=(self.input_units, self.output_units))
        return self

    def forward(self, x):
        self.old_x = x
        return np.dot(x, self.W)

    def backward(self, grad):
        self.grad_W = np.mean(
            np.matmul(self.old_x[:, :, None], grad[:, None, :]), axis=0)
        if self.regularizer is not None:
            self.grad_W += self.regularizer.grad(self.W)
        return np.dot(grad, self.W.transpose())

    def parameters(self):
        return [(self.W, self.grad_W)]

    def compute_output_shape(self):
        return (self.output_units,)

    def regularization_loss(self):
        if self.regularizer is not None:
            return self.regularizer.loss(self.W)
        return 0

    def __str__(self):
        return "Linear(output_units=%d, regularizer=%s)" % (
            self.output_units, str(self.regularizer))


class Bias(Layer):
    def __init__(self, regularizer=None):
        super().__init__()
        self.regularizer = regularizer

    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        self.bias = np.zeros(self.output_shape)
        return self

    def forward(self, x):
        if x.shape[1:] != self.output_shape:
            raise ValueError(
                'Dimension does not match,'
                ' x: %s != output_shape: %s' % (x.shape, self.output_shape))
        self.old_x = x
        return x + self.bias

    def backward(self, grad):
        self.grad_bias = grad.mean(axis=0)
        if self.regularizer is not None:
            self.grad_bias += self.regularizer.grad(self.bias)
        return grad

    def parameters(self):
        return [(self.bias, self.grad_bias)]

    def compute_output_shape(self):
        return self.output_shape

    def regularization_loss(self):
        if self.regularizer is not None:
            return self.regularizer.loss(self.bias)
        return 0

    def __str__(self):
        return "Bias(regularizer=%s)" % str(self.regularizer)


class Dense(Layer):
    def __init__(self, output_units, kernel_regularizer=None,
                 bias_regularizer=None):
        super().__init__()
        self.linear = Linear(output_units, kernel_regularizer)
        self.bias = Bias(bias_regularizer)

    def __call__(self, pre_layer):
        pre_layer = self.linear(pre_layer)
        pre_layer = self.bias(pre_layer)
        return self

    def forward(self, x):
        x = self.linear.forward(x)
        x = self.bias.forward(x)
        return x

    def backward(self, grad):
        grad = self.bias.backward(grad)
        grad = self.linear.backward(grad)
        return grad

    def parameters(self):
        return self.linear.parameters() + self.bias.parameters()

    def compute_output_shape(self):
        return self.bias.compute_output_shape()

    def regularization_loss(self):
        return (
            self.linear.regularization_loss() +
            self.bias.regularization_loss())

    def __str__(self):
        return "Dense[%s, %s]" % (str(self.linear), str(self.bias))


class BatchNormalization(Layer):
    def __init__(self, momentum=0.95):
        super().__init__()
        self.momentum = momentum

    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        self.gamma = np.random.uniform(-0.1, 0.1, self.output_shape)
        self.beta = np.zeros(self.output_shape)
        self.running_mean = 0
        self.running_var = 0
        return self

    def forward(self, x):
        if self.is_testing:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + eps)
            x = self.gamma * x_norm + self.beta
        else:
            self.old_x = x
            self.old_mu = np.mean(x, axis=0)
            self.old_var = np.var(x, axis=0)

            self.x_norm = (x - self.old_mu) / np.sqrt(self.old_var + eps)
            x = self.gamma * self.x_norm + self.beta

            self.running_mean = exp_running_avg(
                self.running_mean, self.old_mu, self.momentum)
            self.running_var = exp_running_avg(
                self.running_var, self.old_var, self.momentum)

        return x

    def backward(self, grad):
        N, _ = self.old_x.shape

        x_mu = self.old_x - self.old_mu
        std_inv = 1. / np.sqrt(self.old_var + eps)

        dX_norm = grad * self.gamma
        dvar = np.sum(dX_norm * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = (np.sum(dX_norm * - std_inv, axis=0) +
               dvar * np.mean(-2. * x_mu, axis=0))

        grad = (dX_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        self.grad_gamma = np.sum(grad * self.x_norm, axis=0)
        self.grad_beta = np.sum(grad, axis=0)

        return grad

    def parameters(self):
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]

    def compute_output_shape(self):
        return self.output_shape

    def __str__(self):
        return "BatchNormalization(momentum=%f)" % self.momentum


class Convolution2D(Layer):
    def __init__(self, filters, size, stride=1, regularizer=None):
        super().__init__()
        assert(size[0] == size[1])
        assert(stride[0] == stride[1])
        self.filters = filters
        self.size = size[0]
        self.stride = stride[0]
        self.regularizer = regularizer

    def __call__(self, pre_layer):
        pre_shape = pre_layer.compute_output_shape()
        assert(len(pre_shape) == 3)
        assert(pre_shape[1] == pre_shape[2])
        input_filters, x_size, _ = pre_shape
        self.W = np.random.uniform(
            low=-np.sqrt(6 / (input_filters + self.filters)),
            high=np.sqrt(6 / (input_filters + self.filters)),
            size=(self.filters, input_filters, self.size, self.size))
        self.b = np.zeros((self.filters, 1))

        if self.stride == 1:
            self.padding = self.size - 1
        else:
            self.padding = self.stride - (x_size - self.size) % self.stride
            self.padding = self.padding % self.stride
        self.x_size_out = ((x_size - self.size + self.padding) // self.stride +
                           1)

        return self

    def forward(self, x):
        batch_num = x.shape[0]

        x_col = im2col_indices(
            x, self.size, self.size, self.padding, self.stride)
        W_col = self.W.reshape(self.filters, -1)
        out = W_col @ x_col + self.b
        out = out.reshape(
            self.filters, self.x_size_out, self.x_size_out, batch_num)
        out = out.transpose(3, 0, 1, 2)

        self.x = x
        self.x_col = x_col

        return out

    def backward(self, grad):
        self.grad_b = np.sum(grad, axis=(0, 2, 3))
        self.grad_b = self.grad_b.reshape(self.filters, -1)

        dout_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        self.grad_W = dout_reshaped @ self.x_col.T
        self.grad_W = self.grad_W.reshape(self.W.shape)
        if self.regularizer is not None:
            self.grad_W += self.regularizer.grad(self.W)
            # self.grad_b += self.regularizer.grad(self.b)

        W_reshape = self.W.reshape(self.filters, -1)
        dX_col = W_reshape.T @ dout_reshaped
        grad = col2im_indices(dX_col, self.x.shape, self.size, self.size,
                              padding=self.padding, stride=self.stride)

        return grad

    def parameters(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]

    def compute_output_shape(self):
        return (self.filters, self.x_size_out, self.x_size_out)

    def regularization_loss(self):
        if self.regularizer is not None:
            return self.regularizer.loss(self.W)
        return 0

    def __str__(self):
        return (
            "Convolution2D(filters=%d, size=[%d, %d]"
            ", stride=[%d, %d], regularizer=%s)" % (
                self.filters, self.size, self.size, self.stride, self.stride,
                str(self.regularizer)))


class AbstractPooling2D(Layer):
    def __init__(self, size=[2, 2], stride=[2, 2]):
        super().__init__()
        assert(size[0] == size[1])
        assert(stride[0] == stride[1])
        self.size = size[0]
        self.stride = stride[0]

    def __call__(self, pre_layer):
        pre_shape = pre_layer.compute_output_shape()
        assert(len(pre_shape) == 3)
        assert(pre_shape[1] == pre_shape[2])
        self.filters = pre_shape[0]
        self.x_size = pre_shape[1]

        if self.stride == 1:
            self.padding = self.size - 1
        else:
            self.padding = (
                self.stride - (self.x_size - self.size) % self.stride)
            self.padding = self.padding % self.stride
        self.x_size_out = (
            (self.x_size - self.size + self.padding) // self.stride + 1)

        return self

    def forward(self, x):
        self.old_x = x
        batch_size = x.shape[0]

        X_reshaped = x.reshape(
            batch_size * self.filters, 1, self.x_size, self.x_size)
        self.X_col = im2col_indices(
            X_reshaped, self.size, self.size, self.padding, self.stride)

        self.max_idx = np.argmax(self.X_col, axis=0)
        self.out = self.X_col[self.max_idx, range(self.max_idx.size)]

        self.out = self.out.reshape(
            self.x_size_out, self.x_size_out, batch_size, self.filters)
        self.out = self.out.transpose(2, 3, 0, 1)

        return self.out

    def backward(self, grad):
        batch_size = self.old_x.shape[0]

        dX_col = np.zeros_like(self.X_col)
        dout_col = grad.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_idx, range(dout_col.size)] = dout_col

        grad = col2im_indices(
            dX_col, (batch_size * self.filters, 1, self.x_size, self.x_size),
            self.size, self.size, padding=self.padding, stride=self.stride)
        grad = grad.reshape(self.old_x.shape)

        return grad

    def compute_output_shape(self):
        return (self.filters, self.x_size_out, self.x_size_out)

    @abc.abstractmethod
    def _forward_pooling(self, X_col):
        return NotImplemented

    @abc.abstractmethod
    def _backward_pooling(self, dX_col):
        return NotImplemented


class Maxpooling2D(AbstractPooling2D):
    def __init__(self, size=[2, 2], stride=[2, 2]):
        super().__init__(size, stride)

    def _forward_pooling(self, X_col):
        self.max_idx = np.argmax(X_col, axis=0)
        out = X_col[self.max_idx, range(self.max_idx.size)]
        return out

    def _backward_pooling(self, dX_col, dout_col):
        dX_col[self.max_idx, range(dout_col.size)] = dout_col
        return dX_col

    def __str__(self):
        return "Maxpooling2D(size=[%d, %d], stride=[%d, %d])" % (
            self.size, self.size, self.stride, self.stride)


class Avgpooling2D(AbstractPooling2D):
    def __init__(self, size=[2, 2], stride=[2, 2]):
        super().__init__(size, stride)

    def _forward_pooling(self, X_col):
        out = np.mean(X_col, axis=0)
        return out

    def _backward_pooling(self, dX_col, dout_col):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    def __str__(self):
        return "Avgpooling2D(size=%d, stride=%d)" % (self.size, self.stride)


class Flatten(Layer):
    def __call__(self, pre_layer):
        pre_shape = pre_layer.compute_output_shape()
        self.output_units = 1
        for dim in pre_shape:
            self.output_units *= dim
        return self

    def forward(self, x):
        self.old_x = x
        x = x.reshape((x.shape[0], -1))
        return x

    def backward(self, grad):
        grad = grad.reshape(self.old_x.shape)
        return grad

    def compute_output_shape(self):
        return (self.output_units,)

    def __str__(self):
        return "Flatten()"


class ReLU(Layer):
    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        return self

    def forward(self, x):
        self.old_x = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)

    def compute_output_shape(self):
        return self.output_shape

    def __str__(self):
        return "ReLU()"


class Sigmoid(Layer):
    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        return self

    def forward(self, x):
        self.old_y = np.exp(x) / (1. + np.exp(x))
        return self.old_y

    def backward(self, grad):
        return self.old_y * (1. - self.old_y) * grad

    def compute_output_shape(self):
        return self.output_shape

    def __str__(self):
        return "Sigmoid()"


class Softmax(Layer):
    def __call__(self, pre_layer):
        self.output_shape = pre_layer.compute_output_shape()
        return self

    def forward(self, x):
        xrel = x - x.max(axis=-1, keepdims=True)
        self.old_y = np.exp(xrel) / np.exp(xrel).sum(axis=1)[:, None]
        return self.old_y

    def backward(self, grad):
        grad = self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])
        assertfinite(grad)
        return grad

    def compute_output_shape(self):
        return self.output_shape

    def __str__(self):
        return "Softmax()"

import numpy as np
from scipy import signal
from dnn import *

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:2000].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

class Convolution():
    def __init__(self, input_shape, kernel_size, n_kernels):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.out_shape = (n_kernels, self.input_shape[1]-self.kernel_size+1, self.input_shape[2]-self.kernel_size+1)
        self.kernels = np.random.randn(*(n_kernels, self.input_shape[0], self.kernel_size, self.kernel_size))
        self.biases = np.random.randn(*(n_kernels, self.out_shape[1], self.out_shape[2]))

    def forward_pass(self, input_img):
        self.input_img = input_img
        self.output = np.zeros((self.n_kernels, self.out_shape[1], self.out_shape[2]))
        for i in range(self.n_kernels):
            for j in range(self.input_shape[0]):
                self.output[i] = signal.correlate2d(self.input_img[j], self.kernels[i, j], 'valid')
        return self.output

    def back_propogate(self, out_error, l_r):
        K_error = np.zeros(self.kernels.shape)
        X_error = np.zeros(self.input_shape)

        for i in range(self.n_kernels):
            for j in range(self.input_shape[0]):
                K_error[i, j] = signal.correlate2d(self.input_img[j], out_error[i], 'valid')
                X_error[j] += signal.correlate2d(out_error[i], self.kernels[i, j], 'full')

        self.kernels -= l_r*K_error
        self.biases -= l_r*out_error
        # return X_error

    def reshape(self, input, output_shape):
        return np.reshape(input, output_shape)

    def fit(self, input_img):
        output = self.forward_pass(input_img)


class DNN():
    def __init__(self, input_size, hl_size):
        self.W1 = np.random.rand(10, 26*26*5) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def feedforward(self, X):
        Z0 = self.W1.dot(X) + self.b1
        A0 = self.relu(Z0)
        Z1 = self.W2.dot(A0) + self.b2
        A1 = self.softmax(Z1)

        return Z0, A0, Z1, A1

    def relu(self, X, derivative=False):
        if derivative:
            return X > 0
        return np.maximum(X, 0)

    def backProp(self, X, Z0, A0, Z1, A1, y_train):
        # y_train = self.one_hot(y_train)

        error = (A1 - y_train)
        change_w2 = np.dot(error, A0.T) / 1000
        error = np.dot(self.W2.T, error) * self.relu(Z0, True)
        change_w1 = np.dot(error, X.T) / 1000
        change_b2 = 1 / m * np.sum(change_w2)
        change_b1 = 1 / m * np.sum(change_w1)

        change_X = np.dot(error.T, self.W1)/ 1000
        return change_w1, change_b1, change_w2, change_b2, change_X

    def update_params(self, dw1, db1, dw2, db2, l_r):
        self.W1 -= l_r * dw1
        self.b1 -= l_r * db1
        self.W2 -= l_r * dw2
        self.b2 -= l_r * db2

    def get_pred(self, A1):
        return np.argmax(A1, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size


if __name__ == '__main__':
    epochs = 1000
    conv_layer = Convolution((1, 28, 28), 3, 5)
    dense_layer = DNN(5*26*26, 100)
    training = True
    for e in range(epochs) and training:
        error = 0

        for x, y in zip(X_train.T, Y_train):
            _x_train = x.reshape(1, 28, 28)
            conv_op = conv_layer.forward_pass(_x_train)
            dens_input = conv_layer.reshape(conv_op, (5*26*26, 1))
            Z0, A0, Z1, A1 = dense_layer.feedforward(dens_input)
            y_op = np.zeros([10, 1])
            y_op[y] = 1
            dw1, db1, dw2, db2, dX = dense_layer.backProp(dens_input, Z0, A0, Z1, A1, y_op)
            dense_layer.update_params(dw1, db1, dw2, db2, 0.1)
            out_error = dX.reshape((5, 26, 26))
            conv_layer.back_propogate(out_error, 0.1)
            predictions = dense_layer.get_pred(A1)
            error += abs(predictions-y)

        print(e)
        print(error)
        # predictions = dense_layer.get_pred(A1)
        # print(dense_layer.get_accuracy(predictions, y_op))


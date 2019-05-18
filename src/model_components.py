import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

from constants import ROW, COLUMN


class InputLayer(Chain):
    def __init__(self,
                 n_input_channel: int,
                 n_output_channel: int,
                 kernel_size: int = 3,
                 pad: int = 1  # padding width
                 ) -> None:
        super(InputLayer, self).__init__()
        with self.init_scope():
            self.cnn = L.Convolution2D(n_input_channel, n_output_channel, ksize=kernel_size, pad=pad)
            self.bn = L.BatchNormalization(n_output_channel)

    def __call__(self,
                 x: chainer.Variable     # (b, n_input_channel, ROW, COLUMN)
                 ) -> chainer.Variable:  # (b, n_output_channel, ROW - kernel_size + 1, COLUMN - kernel_size + 1)
        h = F.dropout(F.relu(self.bn(self.cnn(x))), ratio=0.1)
        return h


class ResNet(Chain):
    def __init__(self,
                 n_input_channel: int,
                 n_output_channel: int,
                 kernel_size: int = 3,
                 pad: int = 1
                 ) -> None:
        super(ResNet, self).__init__()
        with self.init_scope():
            self.cnn1 = L.Convolution2D(n_input_channel, n_output_channel, ksize=kernel_size, pad=pad)
            self.bn1 = L.BatchNormalization(n_output_channel)
            self.cnn2 = L.Convolution2D(n_output_channel, n_output_channel, ksize=kernel_size, pad=pad)
            self.bn2 = L.BatchNormalization(n_output_channel)

    def __call__(self,
                 x: chainer.Variable     # (b, n_output_channel, ROW - kernel_size + 1, COLUMN - kernel_size + 1)
                 ) -> chainer.Variable:  # (b, n_output_channel, ROW - kernel_size + 1, COLUMN - kernel_size + 1)
        h1 = F.relu(self.bn1(self.cnn1(x)))
        h2 = self.bn2(self.cnn2(h1))
        h = F.relu(h2 + x)  # residual
        return h


class OutputLayer(Chain):
    def __init__(self,
                 n_input_channel: int,
                 n_output_channel: int = 2,
                 kernel_size: int = 1
                 ) -> None:
        super(OutputLayer, self).__init__()
        self.n_output_channel = n_output_channel
        self.d_inp = self.n_output_channel * ROW * COLUMN
        self.d_out = ROW * COLUMN + 1
        with self.init_scope():
            self.cnn = L.Convolution2D(n_input_channel, self.n_output_channel, ksize=kernel_size)
            self.bn = L.BatchNormalization(self.n_output_channel)
            # fully connected layer / +1 means adding the choice of passing a turn
            self.fc = L.Linear(self.d_inp, self.d_out)

    def __call__(self,
                 x: chainer.Variable     # (b, n_output_channel, ROW, COLUMN)
                 ) -> chainer.Variable:  # (b, n_output_channel, ROW, COLUMN)
        b = len(x)
        h = F.relu(self.bn(self.cnn(x)))  # (b, n_output_channel, ROW, COLUMN)
        h = h.reshape(b, -1)              # (b, n_output_channel * ROW * COLUMN)
        h = F.dropout(self.fc(h), ratio=0.1)
        return h

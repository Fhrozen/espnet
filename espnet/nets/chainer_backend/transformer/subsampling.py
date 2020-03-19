# encoding: utf-8
"""Class Declaration of Transformer's Input layers."""

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding

import logging
import numpy as np


class Conv2dSubsampling(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(Conv2dSubsampling, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv1 = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         initial_bias=initial_bias(scale=stvd))
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         initial_bias=initial_bias(scale=stvd))
            stvd = 1. / np.sqrt(dims)
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        xs = self.xp.array(xs[:, None])
        xs = F.relu(self.conv1(xs))
        xs = F.relu(self.conv2(xs))
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens


class LinearSampling(chainer.Chain):
    """Linear 1D subsampling.

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize LinearSampling."""
        super(LinearSampling, self).__init__()
        stvd = 1. / np.sqrt(dims)
        self.dropout = dropout
        with self.init_scope():
            self.linear = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                   initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        logging.info(xs.shape)
        xs = self.linear(xs, n_batch_axes=2)
        logging.info(xs.shape)
        xs = self.pe(xs)
        return xs, ilens


class ConvWithNorm(chainer.Chain):
    def __init__(self, in_channels, out_channels,
                 kernel=1, stride=1, pad=0, nobias=True, groups=1):
        super(ConvWithNorm, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, out_channels, kernel, stride=stride, pad=pad, nobias=nobias, groups=groups)
            self.bn = L.GroupNormalization(1, out_channels)

    def __call__(self, x):
        x = self.conv(x)
        return self.bn(x)


class Bottleneck(chainer.Chain):
    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, bn=True, act=F.relu, groups=1):
        super(Bottleneck, self).__init__()
        if bn:
            Conv = ConvWithNorm
        else:
            Conv = L.Convolution2D
        with self.init_scope():
            self.shortcut = Conv(in_channels, out_channels, 1, stride=stride,
                                 pad=0, nobias=True, groups=groups)
            self.conv1 = Conv(in_channels, mid_channels, 3, stride=1, pad=1, nobias=True, groups=groups)
            self.conv2 = Conv(mid_channels, out_channels, 3, stride=stride, pad=1, nobias=True, groups=groups)
        self.act = act
        self.bn = bn

    def __call__(self, x):
        res_x = self.act(self.conv1(x))
        res_x = self.conv2(res_x)
        x = self.shortcut(x)
        return self.act(x + res_x)


class ResBN(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(ResBN, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv0 = L.Convolution2D(1, channels, 1, stride=1, initial_bias=initial_bias(scale=stvd),
                                         nobias=True)
            self.conv1 = Bottleneck(channels, channels, 3, stride=2,
                                         initialW=initialW(scale=stvd))
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv2 = Bottleneck(channels, channels, 3, stride=2,
                                         initialW=initialW(scale=stvd))
            stvd = 1. / np.sqrt(dims)
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        xs = self.xp.array(xs[:, None])
        xs = self.conv0(xs)
        xs = self.conv1(xs)
        xs = self.conv2(xs)
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens


class Residual1(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(Residual1, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv0 = L.Convolution2D(1, channels, 1, stride=1, initial_bias=initial_bias(scale=stvd),
                                         nobias=True)
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv1 = L.Convolution2D(channels, channels, 3, stride=1, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=1, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            stvd = 1. / np.sqrt(dims)
            # logging.info(idim)
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        xs = self.xp.array(xs[:, None])
        # logging.info(xs.shape)
        xs = self.conv0(xs)
        xs = F.relu(self.conv1(xs)) + xs
        # logging.info(xs.shape)
        xs = F.max_pooling_2d(xs, 2, 2)
        # logging.info(xs.shape)
        xs = F.relu(self.conv2(xs)) + xs
        # logging.info(xs.shape)
        xs = F.max_pooling_2d(xs, 2, 2)
        # logging.info(xs.shape)
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens
    


class Residual2(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(Residual2, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv0 = L.Convolution2D(1, channels, 1, stride=1, initial_bias=initial_bias(scale=stvd),
                                         nobias=True)
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv1 = L.Convolution2D(channels, channels, 3, stride=1, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            self.bn1 = L.GroupNormalization(1, channels)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=1, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            self.bn2 = L.GroupNormalization(1, channels)
            stvd = 1. / np.sqrt(dims)
            # logging.info(idim)
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        xs = self.xp.array(xs[:, None])
        # logging.info(xs.shape)
        xs = self.conv0(xs)
        xs = F.relu(self.bn1(self.conv1(xs))) + xs
        # logging.info(xs.shape)
        xs = F.max_pooling_2d(xs, 2, 2)
        # logging.info(xs.shape)
        xs = F.relu(self.bn2(self.conv2(xs))) + xs
        # logging.info(xs.shape)
        xs = F.max_pooling_2d(xs, 2, 2)
        # logging.info(xs.shape)
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens



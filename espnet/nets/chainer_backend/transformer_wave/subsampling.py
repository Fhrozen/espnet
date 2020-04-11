# encoding: utf-8
"""Class Declaration of Transformer's Input layers."""

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer_wave.embedding import PositionalEncoding

import logging
import numpy as np


class WAVConv2dNorm(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(STFTConv2dGN, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            self.norm = L.GroupNormalization(1, idim)
            stvd = 1. / np.sqrt(n)
            self.conv1 = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            stvd = 1. / np.sqrt(dims)

            idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                nobias=True)
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        NFFT = xs.shape[-1]
        xs = 1.0 / NFFT * (np.square(np.absolute(xs)))
        xs[xs <= 1e-30] = 1e-30
        xs = 10 * np.log10(xs)
        # BS x L x D => BS x D x L
        xs = self.xp.array(xs.transpose(0, 2, 1))

        # GN normalization
        xs = F.expand_dims(self.norm(xs).transpose(0, 2, 1), axis=1)
        xs = F.relu(self.conv1(xs))
        xs = F.relu(self.conv2(xs))
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens

class STFTConv2dSub2(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(STFTConv2dSub2, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 6 * 83
            stvd = 1. / np.sqrt(n)
            self.norm = L.GroupNormalization(1, idim)
            self.conv1 = L.Convolution2D(1, dims, [6, 201], stride=4, pad=[1, 0],
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        NFFT = xs.shape[-1]
        xs = 1.0 / NFFT * (np.square(np.absolute(xs)))
        xs[xs <= 1e-30] = 1e-30
        xs = 10 * np.log10(xs)
        # BS x L x D => BS x D x L
        xs = self.xp.array(xs.transpose(0, 2, 1))

        # GN normalization
        xs = F.expand_dims(self.norm(xs).transpose(0, 2, 1), axis=1)
        xs = F.relu(self.conv1(xs)).transpose(0, 2, 1, 3)
        batch, length, _, _ = xs.shape
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens


class STFTConv2dSub3(chainer.Chain):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize Conv2dSubsampling."""
        super(STFTConv2dSub3, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            stvd = 1. / np.sqrt(dims)
            self.norm = L.GroupNormalization(1, idim)
            self.conv1 = L.Convolution2D(1, dims, [4, 201], stride=4, pad=0,
                                         initialW=initialW(scale=stvd),
                                         nobias=True)
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        """Subsample x.

        :param chainer.Variable x: input tensor
        :return: subsampled x and mask

        """
        NFFT = xs.shape[-1]
        xs = 1.0 / NFFT * (np.square(np.absolute(xs)))
        xs[xs <= 1e-30] = 1e-30
        xs = 10 * np.log10(xs)
        # BS x L x D => BS x D x L
        xs = self.xp.array(xs.transpose(0, 2, 1))

        # GN normalization
        xs = F.expand_dims(self.norm(xs).transpose(0, 2, 1), axis=1)
        xs = F.relu(self.conv1(xs)).transpose(0, 2, 1, 3)
        batch, length, _, _ = xs.shape
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens

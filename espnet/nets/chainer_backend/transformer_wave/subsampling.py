# encoding: utf-8
"""Class Declaration of Transformer's Input layers."""

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer_wave.embedding import PositionalEncoding

import logging
import numpy as np


class StftConv2DSubsamp(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None,
                 mels=80, freq_samp=16000, filter_length=512,
                 hop_length=160):
        super(StftConv2DSubsamp, self).__init__()
        import librosa
        # Ref: https://github.com/pseeth/pytorch-stft/blob/master/stft.py#L33
        self.filter_length = filter_length
        self.hop_length = hop_length
        logging.info('Stft param with stft initializer')
        fourier_basis = np.fft.rfft(np.eye(self.filter_length))
        fourier_basis = np.hstack([fourier_basis.real,
                                    fourier_basis.imag]).astype(np.float32).T
        _mel_options = dict(sr=freq_samp,
                            n_fft=filter_length,
                            n_mels=mels,
                            fmin=0.0,
                            fmax=None)
        self.dropout = dropout
        melmat = librosa.filters.mel(**_mel_options).T
        self.fourier_basis = fourier_basis[:, None, None, :]
        self.melmat = melmat.astype(np.float32)
        with self.init_scope():
            self.norm = L.GroupNormalization(1, mels)
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
            idim = int(np.ceil(np.ceil(mels / 2) / 2)) * channels
            self.out = L.Linear(idim, dims, initialW=chainer.initializers.Uniform(scale=stvd),
                            initial_bias=chainer.initializers.Uniform(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def __call__(self, xs, ilens):
        xp = self.xp
        xs = F.expand_dims(xs, axis=1).data
        # BS x 1 x T x NFFT
        xs = F.convolution_2d(xp.array(xs), xp.array(self.fourier_basis), stride=1, pad=0)
        # BS x NFFT/2+1 * 2 x T x 1 
        xs = F.squeeze(xs, axis=3)
        cutoff = int((self.filter_length / 2) + 1)
        real_xs = xs[:, :cutoff, :]
        imag_xs = xs[:, cutoff:, :]
        xs = real_xs ** 2 + imag_xs ** 2

        xs = F.swapaxes(xs, 1, 2)
        xs = F.matmul(xs, xp.array(self.melmat))
        # BS x T x MEL
        xs = F.log(F.absolute(xs) + 1e-20).transpose(0, 2, 1)
        # Norm
        xs = self.norm(xs.data).transpose(0, 2, 1)
        xs = F.expand_dims(xs, axis=1)
        xs = F.relu(self.conv1(xs))
        xs = F.relu(self.conv2(xs))
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
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


class StftRes2DSubsamp(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None,
                 mels=80, freq_samp=16000, filter_length=512,
                 hop_length=160):
        super(StftRes2DSubsamp, self).__init__()
        import librosa
        # Ref: https://github.com/pseeth/pytorch-stft/blob/master/stft.py#L33
        self.filter_length = filter_length
        self.hop_length = hop_length
        logging.info('Stft param with stft initializer')
        fourier_basis = np.fft.rfft(np.eye(self.filter_length))
        fourier_basis = np.hstack([fourier_basis.real,
                                    fourier_basis.imag]).astype(np.float32).T
        _mel_options = dict(sr=freq_samp,
                            n_fft=filter_length,
                            n_mels=mels,
                            fmin=0.0,
                            fmax=None)
        self.dropout = dropout
        melmat = librosa.filters.mel(**_mel_options).T
        self.fourier_basis = fourier_basis[:, None, None, :]
        self.melmat = melmat.astype(np.float32)
        with self.init_scope():
            self.norm = L.GroupNormalization(1, mels)
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv0 = L.Convolution2D(1, channels, 1, stride=1, initial_bias=initial_bias(scale=stvd),
                                         nobias=True)
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv1 = Bottleneck(channels, channels, channels, stride=2,
                                         initialW=initialW(scale=stvd))
            self.conv2 = Bottleneck(channels, channels, channels, stride=2,
                                         initialW=initialW(scale=stvd))
            stvd = 1. / np.sqrt(dims)
            idim = int(np.ceil(np.ceil(mels / 2) / 2)) * channels
            self.out = L.Linear(idim, dims, initialW=chainer.initializers.Uniform(scale=stvd),
                            initial_bias=chainer.initializers.Uniform(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def __call__(self, xs, ilens):
        xp = self.xp
        xs = F.expand_dims(xs, axis=1).data
        # BS x 1 x T x NFFT
        xs = F.convolution_2d(xp.array(xs), xp.array(self.fourier_basis), stride=1, pad=0)
        # BS x NFFT/2+1 * 2 x T x 1 
        xs = F.squeeze(xs, axis=3)
        cutoff = int((self.filter_length / 2) + 1)
        real_xs = xs[:, :cutoff, :]
        imag_xs = xs[:, cutoff:, :]
        xs = real_xs ** 2 + imag_xs ** 2

        xs = F.swapaxes(xs, 1, 2)
        xs = F.matmul(xs, xp.array(self.melmat))
        # BS x T x MEL
        xs = F.log(F.absolute(xs) + 1e-20).transpose(0, 2, 1)
        # Norm
        xs = self.norm(xs.data).transpose(0, 2, 1)
        xs = F.expand_dims(xs, axis=1)
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

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
        xs = F.convolution_2d(xs, xp.array(self.fourier_basis), stride=1, pad=0)
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

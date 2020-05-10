import chainer
from chainer import functions as F
from chainer import links as L

import logging
import numpy as np


class SincNet(chainer.Chain):
    """docstring for SincConv"""
    def __init__(self, out_channels, kernel_size, stride=1, sample_rate=16000,
                 min_low_fq_hz=50., min_band_fq_hz=50., initializer='mel'):
        super(SincNet, self).__init__()
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.min_band_fq_hz = min_band_fq_hz
        self.min_low_fq_hz = min_low_fq_hz
        self.stride = stride
        self.filters = None
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1  # odd kernel
        # Mel-based parameters initializer
        low_hz = 30.
        high_hz = sample_rate / 2 - (min_low_fq_hz + min_band_fq_hz)
        
        if initializer == 'flat':
            filt = np.ones((out_channels)) * low_hz / sample_rate
            bands = np.ones((out_channels)) * 50. / sample_rate
        elif initializer == 'uniform':
            filt = np.sort(np.random.uniform(low_hz, high_hz, out_channels + 1)) / sample_rate
            bands = np.diff(filt)
        else: # mel
            import librosa
            filt = librosa.core.time_frequency.mel_frequencies(out_channels + 1, fmin=low_hz, fmax=high_hz, htk=True) / sample_rate
            bands = np.diff(filt)

        with self.init_scope():
            # filters (out_channels, 1)
            self.low_freq = chainer.Parameter(filt[:-1][:, None].astype(np.float32))
            self.band_freq = chainer.Parameter(bands[:, None].astype(np.float32))
        self.window = np.hamming(self.kernel_size)[None].astype(np.float32)
        self.register_persistent('window')
        # (kernel_size, 1)
        kernel = (self.kernel_size - 1.) / 2.
        self.kernel = 2 * np.pi * np.arange(-kernel, kernel + 1, dtype=np.float32).reshape(1, -1) / sample_rate
        self.register_persistent('kernel')

    def sinc(self, x):
        # Numerically stable definition
        mask = (F.absolute(x).data < 1e-12)
        y = F.where(mask, self.xp.full(x.shape, 1.e-12, dtype= np.float32), x)
        y = F.sin(y) / y
        return y

    def forward(self, xs):
        low = F.absolute(self.low_freq) + self.min_low_fq_hz / self.sample_rate
        high = low + self.min_band_fq_hz / self.sample_rate + F.absolute(self.band_freq)
        high = F.clip(high, self.min_band_fq_hz / self.sample_rate, 0.5)

        low_pass = 2 * low * self.sinc(F.matmul(low, self.kernel) * self.sample_rate)
        high_pass = 2 * high * self.sinc(F.matmul(high, self.kernel) * self.sample_rate)

        band_pass = high_pass - low_pass
        band_pass = (band_pass / F.max(band_pass, axis=1, keepdims=True)) * self.window

        self.filters = band_pass.reshape(self.out_channels, 1, self.kernel_size, 1)
        xs = F.convolution_2d(xs, self.filters, stride=1)
        return xs

import chainer
from chainer import functions as F
from chainer import links as L

import logging
import numpy as np


def to_mel(hz):
    return 2595. * np.log10(1. + hz / 700.)


def to_hz(mel):
    return 700. * (10. ** (mel / 2595.) - 1.)


class SincNet(chainer.Chain):
    """docstring for SincConv"""
    def __init__(self, out_channels, kernel_size, stride=1, sample_rate=16000,
                 min_low_fq_hz=50., min_band_fq_hz=50.):
        super(SincNet, self).__init__()
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.min_band_fq_hz = min_band_fq_hz
        self.min_low_fq_hz = min_low_fq_hz
        self.stride = stride
        self.filters = None
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1  # odd kernel
        # Mel-based parameters initializer
        high_fq_hz = sample_rate / 2 - (min_low_fq_hz + min_band_fq_hz)
        low_hz = 30.
        filt = np.linspace(to_mel(low_hz), to_mel(high_fq_hz), out_channels + 1).astype(np.float32)
        filt = to_hz(filt) / self.sample_rate
        with self.init_scope():
            # filters (out_channels, 1)
            self.low_freq = chainer.Parameter(filt[:-1].reshape(-1, 1))
            self.band_freq = chainer.Parameter(np.diff(filt).reshape(-1, 1))
        self.window = np.hamming(self.kernel_size)[None]
        # (kernel_size, 1)
        kernel = (self.kernel_size - 1.) / 2.
        self.kernel = np.arange(-kernel, kernel + 1).reshape(1, -1) / sample_rate
    
    def sinc(self, x):
        # Numerically stable definition
        x_left = x[:, 0 : int((x.shape[1] - 1) / 2)]
        y_left = F.sin(x_left) / x_left
        y_right = F.flip(y_left, axis=1)
        y = F.concat([y_left, self.xp.ones((x.shape[0], 1), dtype=np.float32), y_right], axis=1)
        return y

    def forward(self, xs, ilens):
        xp = self.xp

        kernel = xp.asarray(self.kernel.astype(np.float32))
        window = xp.asarray(self.window.astype(np.float32))

        low = self.min_low_fq_hz / self.sample_rate + F.absolute(self.low_freq)
        f_times_t_low = F.matmul(low, kernel)
        low_pass = 2 * low * self.sinc(2 * np.pi * f_times_t_low * self.sample_rate)

        high = low + self.min_band_fq_hz / self.sample_rate + F.absolute(self.band_freq)
        f_times_t_high = F.matmul(high, kernel)
        high_pass = 2 * high * self.sinc(2 * np.pi * f_times_t_high * self.sample_rate)

        band_pass = high_pass - low_pass
        band_max = F.max(band_pass, axis=1, keepdims=True)
        band_pass = (band_pass / band_max) * window
        self.filters = F.expand_dims(band_pass, axis=1)
        xs = F.convolution_1d(F.expand_dims(xs, axis=1).data, self.filters, stride=self.stride)
        # xs dims = batch x filters x len
        ilens = ((np.array(ilens, dtype=np.float32) - self.kernel_size) / self.stride).astype(np.int) + 1
        xs = F.log(F.absolute(xs) ** 2 + 1e-20)
        return F.elu(xs), ilens

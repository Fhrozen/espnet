import chainer
from chainer import functions as F
from chainer import links as L

import logging
import numpy as np


def sinc(xp, band, t_right):
    x = 2 * np.pi * band * t_right
    y_right = F.sin(x) / x
    y_left = F.flip(y_right, axis=0)
    y = F.concat([y_left, xp.ones((1,)), y_right], axis=0)
    return y


def to_mel(hz):
    return 2595. * np.log10(1. + hz / 700.)


def to_hz(mel):
    return 700. * (10. ** (mel / 2595.) - 1.)


class SincNet(chainer.Chain):
    """docstring for SincConv"""
    def __init__(self, out_channels, kernel_size, stride=1, sample_rate=16000,
                 min_low_fq_hz=30., min_band_fq_hz=50.):
        super(SincNet, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1  # odd kernel
        # Mel-based parameters initializer
        high_fq_hz = sample_rate / 2 - (min_low_fq_hz + min_band_fq_hz)
        filt = np.linspace(to_mel(min_low_fq_hz), to_mel(high_fq_hz), out_channels + 1).astype(np.float32)
        filt = to_hz(filt)
        with self.init_scope():
            # filters (out_channels, 1)
            self.low_freq = chainer.Parameter(filt[:-1].reshape(-1, 1))
            self.band_freq = chainer.Parameter(np.diff(filt).reshape(-1, 1))

        # Hamming window
        window = np.linspace(0, (self.kernel_size / 2) - 1, num=int(self.kernel_size / 2))  # Half window
        self.window = 0.54 - 0.46 * np.cos(2 * np.pi * window / self.kernel_size)
        
        # (kernel_size, 1)
        kernel = (self.kernel_size - 1.) / 2.
        self.kernel = 2 * np.pi * np.arange(-kernel, 0).reshape(1, -1) / sample_rate  # Half of axes
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.min_band_fq_hz = min_band_fq_hz
        self.min_low_fq_hz = min_low_fq_hz
        self.stride = stride
        self.filters = None
        self.iter = 0

    def __call__(self, xs, ilens):
        xp = self.xp
        self.iter += 1

        kernel = xp.asarray(self.kernel.astype(np.float32))
        window = xp.asarray(self.window.astype(np.float32))

        low = self.min_low_fq_hz + F.absolute(self.low_freq)
        high = F.clip(low + self.min_band_fq_hz + F.absolute(self.band_freq), self.min_low_fq_hz, self.sample_rate / 2.)
        band = (high - low) * 2
        
        f_times_t_low = F.matmul(low, kernel)
        f_times_t_high = F.matmul(high, kernel)

        band_pass_left = ((F.sin(f_times_t_high) - F.sin(f_times_t_low)) / (kernel / 2)) * window
        # band_pass_center = 2 * band
        band_pass_right = F.flip(band_pass_left, axis=1)
        
        band_pass = F.concat([band_pass_left, band, band_pass_right], axis=1) / band
        # band_pass = band_pass 

        self.filters = F.expand_dims(band_pass, axis=1)
        if self.iter > 4000:
            # Cut backprop due to overfitting (?)
            with chainer.no_backprop_mode():
                xs = F.convolution_1d(F.expand_dims(xs, axis=1).data, self.filters, stride=self.stride)
        else:
            xs = F.convolution_1d(F.expand_dims(xs, axis=1).data, self.filters, stride=self.stride)
        # xs dims = batch x filters x len
        ilens = ((np.array(ilens, dtype=np.float32) - self.kernel_size) / self.stride).astype(np.int) + 1
        return F.relu(xs), ilens

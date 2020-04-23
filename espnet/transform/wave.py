# Copied from https://github.com/nttcslab-sp/kaldifeats
from typing import Union, Callable, Optional, Tuple, List

import logging
import numpy
import scipy.fftpack
import scipy.signal
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext


def get_window(window: Union[str, Tuple[str, Union[int, float]]],
               Nx: int, fftbins: bool=True) -> numpy.ndarray:
    """Return a window.
    
    This function depends on scipy.signal.get_window and basically 
    has compatible arguments exception with 
    
    1. 'povey', which is a window function developpend by Dan-povey 
    2. Suport the optional value of coefficiency for "blackman" window
    
    Parameters:
        window (string, float, or tuple):
            The type of window to create. See below for more details.
        Nx (int):
            The number of samples in the window.
        fftbins (bool):
            If True, create a "periodic" window ready to use with ifftshift
            and be multiplied by the result of an fft (SEE ALSO fftfreq).
    Returns:
        get_window (numpy.ndarray):
            Returns a window of length `Nx` and type `window`
        
    Notes:
        Window types:
            povey, 
            boxcar, triang, blackman, hamming, hann, bartlett, flattop,
            parzen, bohman, blackmanharris, nuttall, barthann,
            kaiser (needs beta), gaussian (needs std),
            general_gaussian (needs power, width),
            slepian (needs width), chebwin (needs attenuation)
        If the window requires no parameters, then `window` can be a string.
        If the window requires parameters, then `window` must be a tuple
        with the first argument the string name of the window, and the next
        arguments the needed parameters.
        If `window` is a floating point number, it is interpreted as the beta
        parameter of the kaiser window.
        Each of the window types listed above is also the name of
        a function that can be called directly to create a window of
        that type.
        
    Examples:
        >>> get_window('triang', 7)
        array([ 0.25,  0.5 ,  0.75,  1.  ,  0.75,  0.5 ,  0.25])
        >>> get_window(('kaiser', 4.0), 9)
        array([ 0.08848053,  0.32578323,  0.63343178,  0.89640418,  1.        ,
                0.89640418,  0.63343178,  0.32578323,  0.08848053])
        >>> get_window(4.0, 9)
        array([ 0.08848053,  0.32578323,  0.63343178,  0.89640418,  1.        ,
                0.89640418,  0.63343178,  0.32578323,  0.08848053])
    """
    if isinstance(window, tuple):
        if len(window) == 2 and window[0] in ['blackman', 'black', 'blk']:
            window, coeff = window
            return blackman(Nx, coeff=coeff, sym=fftbins)
    elif window in ['povey', 'danpovey']:
        return povey(Nx, sym=fftbins)
    else:
        return scipy.signal.get_window(window, Nx=Nx, fftbins=fftbins)


def povey(M: int, sym: bool=True) -> numpy.ndarray:
    """
    "povey" is a window dan-povey made to be similar to Hamming 
    but to go to zero at the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
    He just don't think the Hamming window makes sense as a windowing function.
     
    Args:
        M (int)
            Number of points in the output window. If zero or less, an empty
            array is returned.
        sym (bool):
            When True (default), generates a symmetric window, 
            for use in filter design.
            When False, generates a periodic window,
            for use in spectral analysis.
        
    Returns:
        w (numpy.ndarray):
            The window, with the maximum value normalized to 
            1 (though the value 1 does not appear if
             `M` is even and `sym` is True).   
    """
    # Docstring adapted from NumPy's blackman function
    if M < 1:
        return numpy.array([])
    if M == 1:
        return numpy.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M += 1
    n = numpy.arange(0, M)
    w = numpy.power(0.5 - 0.5 * numpy.cos(2.0 * numpy.pi * n / (M - 1)),
                    0.85)
    if not sym and not odd:
        w = w[:-1]
    return w


def blackman(M: int, coeff: float=0.42, sym: bool=True) -> numpy.ndarray:
    """Originated from scipy.signal.windows.blackman 
    with additionally parameter "coeff" for comapatibility of kaldi 
    
    Return a Blackman window.
    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.
    
    Args:
        M (int)
            Number of points in the output window. If zero or less, an empty
            array is returned.
        coeff (float): The coefficient value for blackman function
        sym (bool):
            When True (default), generates a symmetric window, 
            for use in filter design.
            When False, generates a periodic window, 
            for use in spectral analysis.
        
    Returns:
        w (numpy.ndarray):
            The window, with the maximum value normalized to 
            1 (though the value 1 does not appear if
             `M` is even and `sym` is True).
    Notes:
        The Blackman window is defined as
        .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)
        Most references to the Blackman window come from the signal processing
        literature, where it is used as one of many windowing functions for
        smoothing values.  It is also known as an apodization (which means
        "removing the foot", i.e. smoothing discontinuities at the beginning
        and end of the sampled signal) or tapering function. It is known as a
        "near optimal" tapering function, almost as good (by some measures)
        as the Kaiser window.
        References
        ----------
        .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
               spectra, Dover Publications, New York.
        .. [2] Oppenheim, A.V., and R.W. Schafer. 
                Discrete-Time Signal Processing.
               Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
    """
    # Docstring adapted from NumPy's blackman function
    if M < 1:
        return numpy.array([])
    if M == 1:
        return numpy.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = numpy.arange(0, M)
    w = (coeff - 0.5 * numpy.cos(2.0 * numpy.pi * n / (M - 1)) +
         (1 - coeff) * numpy.cos(4.0 * numpy.pi * n / (M - 1)))
    if not sym and not odd:
        w = w[:-1]
    return w


def round_up_to_nearest_power_of_two(n: int) -> int:
    """
    Args:
        n (int):
    Returns:
        N (int)
    Example:
        >>> round_up_to_nearest_power_of_two(123)
        128
        >>> round_up_to_nearest_power_of_two(5)
        8
        >>> round_up_to_nearest_power_of_two(1023)
        1024
    """
    assert n > 0
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def dithering(wave: numpy.ndarray,
              dither_value: float=1.0,
              state_or_seed: Union[numpy.random.RandomState, int]=None)\
        -> None:
    if dither_value == 0.0:
        return
    if state_or_seed is None:
        state = numpy.random.RandomState()
    elif isinstance(state_or_seed, int):
        state = numpy.random.RandomState(state_or_seed)
    else:
        state = state_or_seed
    rand_gauss = numpy.sqrt(-2 * numpy.log(state.uniform(0, 1,
                                                         size=wave.shape))) * \
        numpy.cos(2 * numpy.pi * state.uniform(0, 1, size=wave.shape))
    wave += rand_gauss * dither_value


def pre_emphasis_filter(signal: numpy.ndarray, p=0.97) -> None:
    """Apply pre-emphasis filter to the input array inplace
    
    To implement pre emphasis fitler using scipy.signal.lfilter,
    
    >>> signal = scipy.signal.lfilter([1.0, -p], 1, signal)
        
    and this is equivalent to
    
    >>> signal[..., 1:] -= p * signal[..., :-1]
    
    The process only for the 0 index is different from this function.
    """

    signal[..., 1:] -= p * signal[..., :-1]
    signal[..., 0] -= p * signal[..., 0]


def pre_stft(
        x: numpy.ndarray,
        frame_length: int,
        frame_shift: int,
        window_type: Union[str,
                           Tuple[str, Union[float, int]],
                           numpy.ndarray]='hann',
        nfft: int=None,
        detrend: Union[str,
                       bool,
                       Callable[[numpy.ndarray], numpy.ndarray],
                       None]='constant',
        return_onesided: bool=True,
        boundary: str=None,
        padded: bool=False,
        dither: float=None,
        dither_seed: Union[numpy.random.RandomState, int]=None,
        preemphasis_coefficient: float=0.,
        return_energy: bool=False,
        return_raw_energy: bool=False,
        round_to_power_of_two: bool=True,
        dtype=None,
        deepcopy_input=False,
        ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError('Unknown boundary option "{0}", must be one of: {1}'
                         .format(boundary, list(boundary_funcs.keys())))

    x_org = x
    x = numpy.asarray(x, dtype=dtype)

    if x.size == 0:
        raise ValueError('Input array size is zero')
    if frame_length < 1:
        raise ValueError('frame_length must be a positive integer')
    if frame_length > x.shape[-1]:
        raise ValueError('frame_length is greater than input length')
    if 0 >= frame_shift:
        raise ValueError('frame_shift must be greater than 0')

    # parse window; if array like, then set frame_length = win.shape
    if isinstance(window_type, str) or isinstance(window_type, tuple):
        win = get_window(window_type, frame_length)
    else:
        win = numpy.asarray(window_type)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if x.shape[-1] < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if frame_length != win.shape[0]:
            raise ValueError(
                'value specified for frame_length is '
                'different from length of window')

    if nfft is None:
        nfft = frame_length
    elif nfft < frame_length:
        raise ValueError('nfft must be greater than or equal to frame_length.')
    else:
        nfft = int(nfft)
    if round_to_power_of_two:
        nfft = round_up_to_nearest_power_of_two(nfft)

    if return_onesided and numpy.iscomplexobj(x):
        warnings.warn('Input data is complex, switching to '
                      'return_onesided=False')
        return_onesided = False

    if x.dtype.kind == 'i':
        x = x.astype(numpy.float64)
    if deepcopy_input and x is x_org:
        x = numpy.array(x)
        assert x is not x_org
    del x_org

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, frame_length // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        zeros_shape = x.shape[:-1] + (nadd,)
        x = numpy.concatenate((x, numpy.zeros(zeros_shape, dtype=x.dtype)),
                              axis=-1)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., numpy.newaxis]
    else:
        shape = x.shape[:-1] + \
                ((x.shape[-1] - frame_length) // frame_shift + 1, frame_length)
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = numpy.lib.stride_tricks.as_strided(x, shape=shape,
                                                    strides=strides)
    del x

    if dither is not None and dither != 0.0:
        dithering(result, dither_value=dither,
                  state_or_seed=dither_seed)

    if detrend is not None and detrend:
        if callable(detrend):
            result = detrend(result)
        else:
            assert isinstance(detrend, str)
            result = scipy.signal.signaltools.detrend(result,
                                                      type=detrend, axis=-1)
    if return_raw_energy:
        raw_energy = numpy.sum(result ** 2, axis=1)

    if preemphasis_coefficient is not None \
            and preemphasis_coefficient != 0.0:
        pre_emphasis_filter(result, p=preemphasis_coefficient)

    result = win.astype(dtype) * result
    length, dims = result.shape
    result = numpy.concatenate((result, numpy.zeros((length, nfft - dims), dtype=dtype)), axis=1)
    return result


class WaveFrames(object):
    def __init__(self, nfft, frame_shift, frame_length=None, window_type='povey', preemphasis_coefficient=0.):
        self.nfft = nfft
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.window_type = 'povey'
        self.preemphasis_coefficient = preemphasis_coefficient

    def __repr__(self):
        return ('{name}(nfft={nfft}, frame_shift={frame_shift}, '
                'frame_length={frame_length}, window_type={window_type}, '
                'preemphasis_coefficient={preemphasis_coefficient})'
                .format(name=self.__class__.__name__,
                        nfft=self.nfft,
                        frame_shift=self.frame_shift,
                        frame_length=self.frame_length,
                        window_type=self.window_type,
                        preemphasis_coefficient=self.preemphasis_coefficient))

    def __call__(self, x):
        nfft = self.nfft
        if nfft is None:
            nfft = self.frame_length
        return pre_stft(x,
                 frame_length=self.frame_length,
                 frame_shift=self.frame_shift,
                 window_type=self.window_type,
                 nfft=nfft,
                 detrend='constant',
                 return_onesided=True,
                 boundary=None,
                 padded=False,
                 dtype=numpy.float32,
                 dither=0.,
                 dither_seed=None,
                 preemphasis_coefficient=self.preemphasis_coefficient,
                 return_energy=False,
                 return_raw_energy=False,
                 round_to_power_of_two=True
                 )


class Wave(object):
    def __init__(self, nfft, frame_shift, frame_length=None, window_type='povey', preemphasis_coefficient=0.):
        self.nfft = nfft
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.window_type = 'povey'
        self.preemphasis_coefficient = preemphasis_coefficient

    def __repr__(self):
        return ('{name}(nfft={nfft}, frame_shift={frame_shift}, '
                'frame_length={frame_length}, window_type={window_type}, '
                'preemphasis_coefficient={preemphasis_coefficient})'
                .format(name=self.__class__.__name__,
                        nfft=self.nfft,
                        frame_shift=self.frame_shift,
                        frame_length=self.frame_length,
                        window_type=self.window_type,
                        preemphasis_coefficient=self.preemphasis_coefficient))

    def __call__(self, x):
        x = numpy.asarray(x, dtype=numpy.float32)
        if x.ndim == 1:
            x_max = numpy.amax(x)
            return x[:, None] / x_max
        else:
            # Multichannel
            return x

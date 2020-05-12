from nara_wpe.wpe import wpe


class WPE(object):
    def __init__(self, taps=10, delay=3, iterations=3, psd_context=0,
                 statistics_mode='full', single_channel=False,
                 is_raw=False, size=512, shift=128):
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.psd_context = psd_context
        self.statistics_mode = statistics_mode
        self.single_channel = single_channel
        self.is_raw = is_raw

    def __repr__(self):
        return ('{name}(taps={taps}, delay={delay}'
                'iterations={iterations}, psd_context={psd_context}, '
                'statistics_mode={statistics_mode}, '
                'single_channel={single_channel}, '
                'is_raw={is_raw})'
                .format(name=self.__class__.__name__,
                        taps=self.taps,
                        delay=self.delay,
                        iterations=self.iterations,
                        psd_context=self.psd_context,
                        statistics_mode=self.statistics_mode,
                        single_channel=self.single_channel,
                        is_raw=self.is_raw))

    def __call__(self, xs):
        """Return enhanced

        :param np.ndarray xs: (Time, Channel, Frequency)
        :return: enhanced_xs
        :rtype: np.ndarray

        """
        # nara_wpe.wpe: (F, C, T)
        xs = wpe(xs.transpose((2, 1, 0)),
                 taps=self.taps,
                 delay=self.delay,
                 iterations=self.iterations,
                 psd_context=self.psd_context,
                 statistics_mode=self.statistics_mode)
        return xs.transpose(2, 1, 0)

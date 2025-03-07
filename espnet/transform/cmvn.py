"""CMVN module."""

import io

import h5py
import kaldiio
import numpy as np


class CMVN(object):
    """CMVN class."""

    def __init__(
        self,
        stats,
        norm_means=True,
        norm_vars=False,
        filetype="mat",
        utt2spk=None,
        spk2utt=None,
        reverse=False,
        std_floor=1.0e-20,
    ):
        """Initialize class."""
        self.stats_file = stats
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.reverse = reverse

        if isinstance(stats, dict):
            stats_dict = dict(stats)
        else:
            # Use for global CMVN
            if filetype == "mat":
                stats_dict = {None: kaldiio.load_mat(stats)}
            # Use for global CMVN
            elif filetype == "npy":
                stats_dict = {None: np.load(stats)}
            # Use for speaker CMVN
            elif filetype == "ark":
                self.accept_uttid = True
                stats_dict = dict(kaldiio.load_ark(stats))
            # Use for speaker CMVN
            elif filetype == "hdf5":
                self.accept_uttid = True
                stats_dict = h5py.File(stats)
            else:
                raise ValueError("Not supporting filetype={}".format(filetype))

        if utt2spk is not None:
            self.utt2spk = {}
            with io.open(utt2spk, "r", encoding="utf-8") as f:
                for line in f:
                    utt, spk = line.rstrip().split(None, 1)
                    self.utt2spk[utt] = spk
        elif spk2utt is not None:
            self.utt2spk = {}
            with io.open(spk2utt, "r", encoding="utf-8") as f:
                for line in f:
                    spk, utts = line.rstrip().split(None, 1)
                    for utt in utts.split():
                        self.utt2spk[utt] = spk
        else:
            self.utt2spk = None

        # Kaldi makes a matrix for CMVN which has a shape of (2, feat_dim + 1),
        # and the first vector contains the sum of feats and the second is
        # the sum of squares. The last value of the first, i.e. stats[0,-1],
        # is the number of samples for this statistics.
        self.bias = {}
        self.scale = {}
        for spk, stats in stats_dict.items():
            assert len(stats) == 2, stats.shape

            count = stats[0, -1]

            # If the feature has two or more dimensions
            if not (np.isscalar(count) or isinstance(count, (int, float))):
                # The first is only used
                count = count.flatten()[0]

            mean = stats[0, :-1] / count
            # V(x) = E(x^2) - (E(x))^2
            var = stats[1, :-1] / count - mean * mean
            std = np.maximum(np.sqrt(var), std_floor)
            self.bias[spk] = -mean
            self.scale[spk] = 1 / std

    def __repr__(self):
        """Return a printable representation of the class."""
        return (
            "{name}(stats_file={stats_file}, "
            "norm_means={norm_means}, norm_vars={norm_vars}, "
            "reverse={reverse})".format(
                name=self.__class__.__name__,
                stats_file=self.stats_file,
                norm_means=self.norm_means,
                norm_vars=self.norm_vars,
                reverse=self.reverse,
            )
        )

    def __call__(self, x, uttid=None):
        """Process call method."""
        if self.utt2spk is not None:
            spk = self.utt2spk[uttid]
        else:
            spk = uttid

        if not self.reverse:
            if self.norm_means:
                x = np.add(x, self.bias[spk])
            if self.norm_vars:
                x = np.multiply(x, self.scale[spk])

        else:
            if self.norm_vars:
                x = np.divide(x, self.scale[spk])
            if self.norm_means:
                x = np.subtract(x, self.bias[spk])

        return x


class UtteranceCMVN(object):
    """Utterance CMVN class."""

    def __init__(self, norm_means=True, norm_vars=False, std_floor=1.0e-20):
        """Initialize class."""
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.std_floor = std_floor

    def __repr__(self):
        """Return a printable representation of the class."""
        return "{name}(norm_means={norm_means}, norm_vars={norm_vars})".format(
            name=self.__class__.__name__,
            norm_means=self.norm_means,
            norm_vars=self.norm_vars,
        )

    def __call__(self, x, uttid=None):
        """Process call method."""
        # x: [Time, Dim]
        square_sums = (x**2).sum(axis=0)
        mean = x.mean(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)

        if self.norm_vars:
            var = square_sums / x.shape[0] - mean**2
            std = np.maximum(np.sqrt(var), self.std_floor)
            x = np.divide(x, std)

        return x

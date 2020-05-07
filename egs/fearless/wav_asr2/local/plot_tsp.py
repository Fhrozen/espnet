#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
from sys import stdout

import numpy as np

from scipy.signal import chirp

import chainer
from chainer import functions as F

from matplotlib import pyplot as plt

from espnet.transform.transformation import Transformation

def prepare_chirp(fs=8000):
    secs=4
    t = np.linspace(0, secs, secs * fs + 1)
    x = chirp(t, f0=fs/2, f1=0, t1=secs, method='linear')
    return x.astype(np.float32)

def process_mels(xs, melmat, fourier_basis, filter_length):
    xs = F.expand_dims(xs, axis=1).data
    # BS x 1 x T x NFFT
    xs = F.convolution_2d(xs, fourier_basis, stride=1, pad=0)
    # BS x NFFT/2+1 * 2 x T x 1 
    xs = F.squeeze(xs, axis=3)
    cutoff = int((filter_length / 2) + 1)
    real_xs = xs[:, :cutoff, :]
    imag_xs = xs[:, cutoff:, :]
    xs = real_xs ** 2 + imag_xs ** 2

    xs = F.swapaxes(xs, 1, 2)
    xs = F.matmul(xs, melmat)
    # BS x T x MEL
    xs = F.log(F.absolute(xs) + 1e-20).transpose(0, 2, 1)
    return xs.data


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    print("snapshots", len(last))
    os.makedirs(args.out, exist_ok=True)
 
    xs = prepare_chirp()
    if not args.preprocess_conf is None:
        preprocessing = Transformation(args.preprocess_conf)
        xs = preprocessing(xs)
    filter_length = xs.shape[1]
    xs = xs[None]
    
    for path in last:
        states = np.load(path)
        snap = path.split('snapshot.')[1]

        fourier_basis = [x for x in states if (('fourier_basis' in x) and ('model' in x))][0]
        melmat = [x for x in states if (('melmat' in x) and ('model' in x))][0]

        fourier_basis = states[str(fourier_basis)]
        melmat = states[str(melmat)]

        mels = process_mels(xs, melmat, fourier_basis, filter_length)[0]

        plt.imshow(mels, origin='bottom')
        plt.savefig(os.path.join(args.out, f'chirp.{snap}.png'))
        
        plt.close()


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    parser.add_argument("--preprocess-conf", default=None, type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
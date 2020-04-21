#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
from sys import stdout

import numpy as np

import chainer

from espnet.nets.chainer_backend.transformer_wave.sincnet import SincNet

from matplotlib import pyplot as plt


def get_filt(filters):
    filt_sum = 0
    for i in range(filters.shape[0]):
        this_filt = filters[i]
        filt_fft = np.fft.rfft(this_filt)
        filt_fft = np.absolute(filt_fft)
        filt_sum += filt_fft
    return filt_sum / np.amax(filt_sum)


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    print("snapshots", len(last))
    os.makedirs(args.out, exist_ok=True)

    with chainer.no_backprop_mode(), chainer.using_config('train', False): 
        x = np.zeros((1, 256), dtype=np.float32)
        x[0, 0] = 1
        net_feats = SincNet(80, 256, stride=80, sample_rate=8000)
        _, _ = net_feats(x, [256])
        norm_filter = get_filt(net_feats.filters.data[:,0])
        plt.plot(norm_filter)
        plt.xlim(0, 129)
        plt.savefig(os.path.join(args.out, 'filters.ep.0.png'))
        plt.close()
        for path in last:
            states = np.load(path)
            snap = path.split('model.')[1]

            band_freq = [x for x in states if 'band_freq' in x][0]
            low_freq = [x for x in states if 'low_freq' in x][0]

            net_feats.band_freq.data = states[str(band_freq)]
            net_feats.band_freq.low_freq = states[str(low_freq)]
            _, _ = net_feats(x, [256])
            norm_filter = get_filt(net_feats.filters.data[:,0])
            plt.plot(norm_filter)
            plt.xlim(0, 129)
            plt.savefig(os.path.join(args.out, f'filters.{snap}.png'))
            
            plt.close()
            net_feats = SincNet(80, 256, stride=80, sample_rate=8000)


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
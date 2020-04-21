#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from sys import stdout

import numpy as np


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    print("snapshots", len(last))
    os.makedirs(args.out, exist_ok=True)

    for path in last:
        stdout.write(f'{path}\r')
        stdout.flush()
        snap = path.split('snapshot.')[1]
        fileout = os.path.join(args.out, f'model.{snap}')
        states = np.load(path)
        keys = [x.split('main/')[1] for x in states if 'model' in x]
        new_keys = dict()
        for k in keys:
            new_keys[k] = states['updater/model:main/{}'.format(k)]

        np.savez_compressed(fileout, **new_keys)
        os.rename('{}.npz'.format(fileout), fileout)  # numpy save with .npz extension


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
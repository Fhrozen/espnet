#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np


def main():
    states = np.load(args.snapshot)
    print(args.snapshot)
    keys = [x.split('input_layer/')[1] for x in states if 'input' in x]
    new_key = dict()
    for k in keys:
        new_key[k] = states['encoder/input_layer/{}'.format(k)]
    np.savez_compressed(args.out, **new_key)
    os.rename('{}.npz'.format(args.out), args.out)  # numpy save with .npz extension


def get_parser():
    parser = argparse.ArgumentParser(description='get input layer')
    parser.add_argument("--snapshot", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()

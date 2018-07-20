#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import os
import sys
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Transcription file')
    parser.add_argument('--outfolder', type=str, help='Output folder')
    parser.add_argument('--utters', type=int, help='Number of Utterances')
    args = parser.parse_args()

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    logging.debug("reading %s", args.text)
    with open(args.text, 'rt', encoding="utf-8") as f:
        lines = f.read().split('\n')

    lines = [x for x in lines if len(x)>1]

    if len(lines) <= args.utters:
        raise ValueError('The number of the subset is larger than the inputted file lenght')

    idx = np.sort(np.random.permutation(len(lines))[: args.utters])
    new_lines = [lines[x] for x in idx]
    new_lines = '\n'.join(new_lines)

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)
    sys.stdout = open(os.path.join(args.outfolder, 'text'), "w+", encoding="utf-8")
    print(new_lines)
    sys.stdout.close()

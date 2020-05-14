#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse

def main():
    for _file in args.results:
        with open(_file, 'r') as f:
            text = f.read().split("\n")
        new_text = list()
        for line in text:
            try:
                _txt, utt = line.split('(')
            except Exception as e:
                break
            
            utt = utt.split(')')[0].split('FS02_')[1]
            utt = utt.replace('TRACK2_DEV', 'track2_dev')
            utt = utt.replace('TRACK2_EVAL', 'track2_eval')
            _txt = _txt.replace(' ', '').replace('<space>', ' ')
            new_text.append(f'FS02_{utt} {_txt}')
        new_text = sorted(new_text)
        dirname = os.path.dirname(_file)
        with open(os.path.join(dirname, 'transcription'), 'w') as f:
            f.write('\n'.join(new_text))


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--results", required=True, type=str, nargs="+")
    parser.add_argument("--out", default=None, type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()



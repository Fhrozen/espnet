#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function

import argparse
import json
import logging
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    args = parser.parse_args()
    
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    new_dic = dict()
    for json in args.jsons:
        j = json.load(open(args.json))
        for item in j['utts'].items():
            id, dic = item
            new_dic[id] = dic
    
    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8')
    print(jsonstring)

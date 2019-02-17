#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--multi', '-m', type=int,
                        help='Test the json file for multiple input/output', default=0)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--output-json', default='', type=str,
                        help='output json file')
    args = parser.parse_args()
    
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # make intersection set for utterance keys
    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info('new json has ' + str(len(intersec_ks)) + ' utterances')
        
    old_dic = dict()
    for k in intersec_ks:
        v = js[0]['utts'][k]
        for j in js[1:]:
            v.update(j['utts'][k])
        old_dic[k] = v

    new_dic = dict()
    if args.multi == 0:
        # single input single output
        for id in old_dic:
            dic = old_dic[id]

            in_dic = {}
            if dic.has_key(unicode('idim', 'utf-8')):
                in_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('ilen', 'utf-8')]), int(dic[unicode('idim', 'utf-8')]))
            in_dic[unicode('name', 'utf-8')] = unicode('input1', 'utf-8')
            in_dic[unicode('feat', 'utf-8')] = dic[unicode('feat', 'utf-8')]

            out_dic = {}
            out_dic[unicode('name', 'utf-8')] = unicode('target1', 'utf-8')
            out_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('olen', 'utf-8')]), int(dic[unicode('odim', 'utf-8')]))
            out_dic[unicode('text', 'utf-8')] = dic[unicode('text', 'utf-8')]
            out_dic[unicode('token', 'utf-8')] = dic[unicode('token', 'utf-8')]
            out_dic[unicode('tokenid', 'utf-8')] = dic[unicode('tokenid', 'utf-8')]

            new_dic[id] = {unicode('input', 'utf-8'):[in_dic], unicode('output', 'utf-8'):[out_dic],
                unicode('utt2spk', 'utf-8'):dic[unicode('utt2spk', 'utf-8')]}
    elif args.multi == 1:
        # multiple input single output
        items = ['.'.join([item.split('.')[0], '-'.join(item.split('-')[1:])]) for item in old_dic]
        items = sorted(list(set(items)))
        items = [x.split('.') for x in items]

        list_inputs = [x.split('.')[1].split('-')[0] for x in old_dic]
        list_inputs = sorted(list(set(list_inputs)))

        for item in items:
            uttr, lapse = item
            in_dics = list()

            for _input in list_inputs:
                if lapse == '':
                    id = '{}.{}'.format(uttr, _input)
                else:
                    id = '{}.{}-{}'.format(uttr, _input, lapse)
                if id in old_dic:
                    dic = old_dic[id]
                    in_dic = dict()
                    if dic.has_key(unicode('idim', 'utf-8')):
                        in_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('ilen', 'utf-8')]), int(dic[unicode('idim', 'utf-8')]))
                    in_dic[unicode('name', 'utf-8')] = _input
                    in_dic[unicode('feat', 'utf-8')] = dic[unicode('feat', 'utf-8')]
                    in_dics.append(in_dic)

            # using last dic
            out_dic = {}
            out_dic[unicode('name', 'utf-8')] = unicode('target1', 'utf-8')
            out_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('olen', 'utf-8')]), int(dic[unicode('odim', 'utf-8')]))
            out_dic[unicode('text', 'utf-8')] = dic[unicode('text', 'utf-8')]
            out_dic[unicode('token', 'utf-8')] = dic[unicode('token', 'utf-8')]
            out_dic[unicode('tokenid', 'utf-8')] = dic[unicode('tokenid', 'utf-8')]

            if lapse == '':
                id = '{}.{}CHs'.format(uttr, len(in_dics))
            else:
                id = '{}.{}CHs-{}'.format(uttr, len(in_dics), lapse)
            new_dic[id] = {unicode('input', 'utf-8'):in_dics, unicode('output', 'utf-8'):[out_dic],
                unicode('utt2spk', 'utf-8'):dic[unicode('utt2spk', 'utf-8')]}    
    elif args.multi == 2:
        # single input multiple output
        raise ValueError('WIP')  
    elif args.multi == 3:
        # multiple input multiple output
        raise ValueError('WIP')
    else:
        raise ValueError('Wrong preparation index')    

    # ensure "ensure_ascii=False", which is a bug
    if args.output_json:
        with codecs.open(args.output_json, "w", encoding='utf-8') as json_file:
            json.dump({'utts': new_dic}, json_file, indent=4, ensure_ascii=False, sort_keys=True, encoding="utf-8")
    else:
        sys.stdout = codecs.getwriter('utf8')(sys.stdout)
        json.dump({'utts': new_dic}, sys.stdout, indent=4, ensure_ascii=False, sort_keys=True, encoding="utf-8")

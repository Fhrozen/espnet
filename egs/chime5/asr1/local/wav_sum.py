#!/usr/bin/python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import numpy as np
import sys
import soundfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, help='Source file')
    parser.add_argument('--list_channels', '-c', type=str, help='Channels to sum')
    parser.add_argument('--source_dir', type=str, help='Source directory')
    parser.add_argument('--result_dir', type=str, help='Result directory')
    args = parser.parse_args()

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    logging.debug("reading %s", args.source)

    with open(args.list_channels) as f:
        listfiles = f.readlines()
    listfiles = [ x.split('\n')[0] for x in listfiles]
    listfiles = [x for x in listfiles if args.source in x][0]
    listfiles = listfiles.split(' ')[1:]
    
    for ch in listfiles:
        filename = '{}/{}'.format(args.source_dir, ch)
        print('Opening file {}'.format(filename))
        data, frqsmp = soundfile.read(filename)
        if not 'data_array' in locals():
            data_array = np.array(data, copy=True)
        else:
            if data.shape[0] > data_array.shape[0]:
                data_array += data[: data_array.shape[0]]
            else:
                data_array[: data.shape[0]] += data
    filename='{}/{}.wav'.format(args.result_dir, args.source)
    print('Opening file {}'.format(filename))
    print('NumChannels is {}, sampling rate is {}'.format(len(listfiles), frqsmp))
    data_array = data_array/np.amax(np.absolute(data_array))
    soundfile.write(filename, data_array, frqsmp)
    

    

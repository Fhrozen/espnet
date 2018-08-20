#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from cStringIO import StringIO

import numpy as np
from python_speech_features import fbank
from python_speech_features import sigproc
import re
from scipy.io import wavfile
import struct
import sys


def get_key(data):
    key = ''
    i = 0
    while 1:
        char = data[i].decode("latin1")
        i += 1
        if char == '' : break
        if char == ' ' : break
        key += char
        
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key, i


def spectrogram(signal, samplerate, winlen=0.025, winstep=0.01, preemph=0.97, nfft=512, winfunc=lambda x:np.hamming(x)):
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    complex_spec = np.fft.rfft(frames, nfft)
    real_feat = np.real(complex_spec).astype(np.float32)
    imag_feat = np.real(complex_spec).astype(np.float32)
    feats = np.concatenate((real_feat, imag_feat), axis=0)
    return feats

def write_mat(key, m):
    fd = ''
    if key != '' : fd += (key+' ').encode("latin1") # ark-files have keys (utterance-id),
    fd += '\0B'.encode() # we write binary!
    # Data-type,
    if m.dtype == 'float32': 
        fd += 'FM '.encode()
    elif m.dtype == 'float64': 
        fd += 'DM '.encode()
    else: 
        raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
    # Dims,
    fd += '\04'.encode()
    fd += struct.pack(np.dtype('uint32').char, m.shape[0]) # rows
    fd += '\04'.encode()
    fd += struct.pack(np.dtype('uint32').char, m.shape[1]) # cols
    # Data,
    fd += m.tobytes()
    sys.stdout.write(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-mel-bins', type=int, help='Mel bins', default=80)
    parser.add_argument('--frame-lenght', type=float, help='Frame lenght (ms)', default=25)
    parser.add_argument('--frame-shift', type=float, help='Frame shift (ms)', default=10)
    parser.add_argument('--preemph', type=float, help='Preemphasis filter', default=0.97)
    parser.add_argument('--use-log-bank', type=int, help='Return Spectrogram if 0, log filterbank feats otherwise', default=0)
    args = parser.parse_args()

    data = sys.stdin.read()
    i = 0
    max_range = 16000 * 3 * 60 * 2
    while i < len(data):
        try:
            key, _i = get_key(data[i:i+120])
            i += _i
            sio =  StringIO(data[i:i + max_range])
            fs, new_data = wavfile.read(sio)
            i += new_data.shape[0]*2 + 44
            if args.use_log_bank < 1:
                feats = spectrogram(new_data, fs,  winlen=0.001 * args.frame_lenght, winstep=0.001 * args.frame_shift,
                    preemph=args.preemph, winfunc=lambda x:np.hamming(x))
            else:
                feats, _ = fbank(new_data, samplerate=fs, winlen=0.001 * args.frame_lenght, winstep=0.001 * args.frame_shift,
                    preemph=args.preemph, nfilt=args.num_mel_bins, winfunc=lambda x:np.hamming(x))
                #feats = np.concatenate((fbanks.astype(np.float32), energy[:, None].astype(np.float32)), axis=1)
            # print(key)
            # with h5py.File('{}.h5'.format(key), 'w') as f:
            #    f.create_dataset('data', data=feats)
            write_mat(key, feats)
        except Exception as e:
            print(e)
            i += 1 
            pass


    
#!/usr/bin/env python


import argparse
import logging
import glob
import os
import platform
import random
import multiprocessing as mp
import six
import sys
import shutil

import numpy as np
from numpy import linalg as LA

import soundfile as sf


def add_noise(_AUDIO, _NOISE='clean', _SNR=None):

    """_AUDIO: Array of multiple channels audio with dimension SxC

               S: # of Samples

               C: # of Channels

       _NOISE: file or color name of the noise

       _SNR: Signal to Noise rate in dB

    """

    if not ((_NOISE is None) or (_SNR is None) or (_NOISE.lower() == 'clean')):
        color = ['white']  # TODO(nelson): Add more color noises
        _LEN = _AUDIO.shape[0]
        dims = len(_AUDIO.shape)
        if dims == 2:
            _CHNS = _AUDIO.shape[1]
        else:
            _CHNS = 0

        if os.path.exists(_NOISE):
            noise, _ = sf.read(_NOISE)
            noise /= np.amax(np.abs(noise))
            if noise.shape[0] < _LEN:
                logging.warning(
                    """The noise file lenght is shorter than the audio file,
                    repeting the noise file to fit the audio file...""")
                rep = int(np.ceil(_LEN / noise.shape[0]) + 1)
                noise = np.tile(noise, rep)
            ii32 = np.iinfo(np.int32)
            max_range = np.amin((ii32.max - 1, noise.shape[0] - _LEN))  # Int32 Protection
            _start = np.random.randint(0, max_range)
            noise = noise[_start:_start + _LEN]
            noise = noise.astype(np.float) / np.amax(np.absolute(noise.astype(np.float)))
            if _CHNS != 0:
                if len(noise.shape) < 2:
                    noise = np.tile(noise[:, np.newaxis], [1, _CHNS])
                elif noise.shape[1] != _CHNS:
                    logging.error(
                        'The noise file should have one channel or a same number of channel as the input...')
                    raise ImportError(
                        'The noise file should have one channel or a same number of channel as the input...')

        elif _NOISE.lower() in color:
            _NOISE = _NOISE.lower()

            if _NOISE == 'white':
                noise = np.random.uniform(-1.0, 1.0, (_AUDIO.shape))
        else:
            logging.error('{} can not be found in or implemented. Please check the available noises'.format(_NOISE))
            raise ValueError('{} can not be found in or implemented. Please check the available noises'.format(_NOISE))

        # TODO(nelson: This will be cutted into frame lenghts to resample the noise
        noise = noise / LA.norm(noise) * LA.norm(_AUDIO) / np.power(10, 0.05 * float(_SNR))
        _AUDIO = _AUDIO + noise
        del noise
    _max = np.amax(np.absolute(_AUDIO))
    if _max > 1.0:
        _AUDIO /= np.amax(np.absolute(_AUDIO))  # Audio Normalized to 1 when it is higher.
    return _AUDIO


def donoisy(arguments):
    args, audio, noisefile = arguments
    snr = np.random.randint(7, 12)
    audio_out = audio.replace(args.audio_folder, args.folder)
    inwave, _ = sf.read(audio)
    inwave = inwave / np.amax(np.absolute(inwave))
    outwave = add_noise(inwave, noisefile, snr)
    sf.write(audio_out, outwave, 16000)


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--folder', default=None, type=str,
                        help='Folder to save')
    parser.add_argument('--noise-type', default=None, type=str,
                        help='Noise File/Type')
    parser.add_argument('--audio-folder', default=None, type=str,
                        help='Audio Folder')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity')
    args = parser.parse_args()

    pool = mp.Pool(processes=10)
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    if os.path.exists(args.noise_type):
        # Formatting Noise
        fn = os.path.basename(args.noise_type)
        outdir = './tmp'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        noisefile = os.path.join(outdir, fn)
        os.system('sox {} -c 1 -r 16000 {}'.format(args.noise_type, noisefile))
    else:
        noisefile = args.noise_type

    audiolist = glob.glob('{}/*.wav'.format(args.audio_folder))

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    thislist = ([args, x, noisefile] for x in audiolist)
    pool.map(donoisy, thislist)

    if os.path.exists(args.noise_type): 
        shutil.rmtree('./tmp')


if __name__ == '__main__':
    main()
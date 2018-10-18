#!/usr/bin/env python3.6

# Copyright 2018 Johns Hopkins University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import logging

from nara_wpe import project_root
from nara_wpe.utils import stft
from nara_wpe.utils import istft
from nara_wpe.utils import get_stft_center_frequencies
from nara_wpe.wpe import online_wpe_step
from nara_wpe.wpe import get_power_online
from nara_wpe.wpe import wpe

import numpy as np
import os
from tqdm import tqdm
from sys import stdout
import soundfile as sf


def online(signal, options, taps, alpha, delay):
    raise Exception('WIP')
    channels = signal.shape[0]
    frequency_bins = options['size'] // 2 + 1
    Z_list = []
    Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
    G = np.zeros((frequency_bins, channels * taps, channels))

    buffer_step = options['shift'] * (taps + delay - 2)

    for i in tqdm(range(0, signal.shape[1], buffer_step)):
        #logging.info(buffer_step)
        y_step = signal[:, i:i + (buffer_step + 2)]
        #logging.info(y_step.shape)
        Y_step = stft(y_step, **options).transpose(1, 2, 0)
        #logging.info(Y_step.shape)
        #exit()
        Z, Q, G = online_wpe_step(Y_step, get_power_online(
            Y_step.transpose(1, 2, 0)), Q, G, alpha=alpha, taps=taps, delay=delay)
        Z_list.append(Z)
        if len(Z_list) > 7500:
            break

    Z_stacked = np.stack(Z_list)
    z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=options['size'], shift=options['shift'])
    return z


def offline(signal, options, iterations, length, frequency):
    out_signal = np.zeros(signal.shape)
    step = length * 60 * frequency
    for i in tqdm(range(0, signal.shape[1], step)):
        signal_step = signal[:, i : i + step]
        Y = stft(signal_step, **options).transpose(2, 0, 1)
        Z = wpe(Y, iterations=iterations, statistics_mode='full').transpose(1, 2, 0)
        z_np = istft(Z, size=options['size'], shift=options['shift'])
        this_length = np.amin((out_signal[:, i : i + step].shape[1], z_np.shape[1]))
        out_signal[:, i : i + this_length] = z_np[:, :this_length]  # Remove pads
    return out_signal


def main(args):
    stft_options = dict(
        size=args.wpe_size,
        shift=args.wpe_shift,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False
    )
    basename = os.path.basename(args.filename)
    logging.info('Loading file: {}'.format(args.filename))
    signal_list = [
        sf.read(args.filename.replace('.CH1', '.CH{}'.format(d)))[0]
        for d in range(1, args.num_channels + 1)
    ]
    audio = np.stack(signal_list, axis=0)
    if args.online:
        logging.info('Initializing WPE online')
        wave_out = online(audio, stft_options, args.taps, args.alpha, args.delay)
    else:
        logging.info('Initializing WPE offline')
        wave_out = offline(audio, stft_options, args.iterations, args.wave_max_length, args.sampling_rate)

    for i in range(1, args.num_channels + 1):
        outfile_name = os.path.join(args.save_folder, basename.replace('.CH1', '.CH{}'.format(i)))
        logging.info('Writing output file at {}'.format(outfile_name))
        sf.write(outfile_name, wave_out[i - 1], args.sampling_rate)
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str, help='Filename of the first channel')
    parser.add_argument('--save-folder',
                        type=str, help='Folder to save file')
    parser.add_argument('--source', '-s',
                        type=str, help='Source file')
    parser.add_argument('--num-channels', '-c',
                        type=int, help='Number of channels', default=4)
    parser.add_argument('--wpe-size',
                        type=int, help='WPE window size', default=512)
    parser.add_argument('--wpe-shift',
                        type=int, help='WPE shift size', default=128)
    parser.add_argument('--online',
                        type=int, help='Online Processing', default=0)
    parser.add_argument('--wave-max-length',
                        type=int, help='Maximum length to wave process (in mins)', default=1)
    parser.add_argument('--sampling-rate',
                        type=int, help='Sampling frequency', default=16000)
    parser.add_argument('--delay',
                        type=int, help='WPE delay', default=3)
    parser.add_argument('--iterations',
                        type=int, help='WPE iterations', default=5)
    parser.add_argument('--alpha',
                        type=float, help='WPE online alpha', default=0.9999)
    parser.add_argument('--taps',
                        type=int, help='WPE taps', default=10)
    args = parser.parse_args()
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    main(args)

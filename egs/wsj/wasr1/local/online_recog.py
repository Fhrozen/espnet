#!/usr/bin/env python3

import logging
import random
import sys

import numpy as np
import signal
import soundfile as sf

import sounddevice as sd
import time

from espnet.bin.asr_recog import get_parser

# chainer related
import chainer

from chainer import training

from chainer.datasets import TransformDataset
from chainer.training import extensions

from scipy.signal import resample

# espnet related
from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.transform.transformation import Transformation


def signal_handler(signal, frame):
    logging.warning('Finish Decoding... ')
    sys.exit(0)


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    set_deterministic_chainer(args)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from ' + args.model)
    # To be compatible with v.0.3.0 models
    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.chainer_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    chainer_load(args.model, model)
    
    logging.info('ARGS: preprocess_conf: ' + str(train_args.preprocess_conf))
    preprocessing = Transformation(train_args.preprocess_conf)

    logging.info('recording Started...')
    # decode each utterance

    duration = 10.5  # seconds
    fs = 48000
    sd.default.samplerate = fs
    sd.default.device = 'Microcone'
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    logging.info('recording finished')
    fs = 16000
    myrecording = resample(myrecording, int(fs * duration))[:, 0]
    sf.write('test.wav', myrecording, fs)
    feat = preprocessing([myrecording])
    with chainer.no_backprop_mode():
        logging.info(f'decoding {feat[0].shape}')
        nbest_hyps = model.recognize(feat[0], args, train_args.char_list, None)
    text = [train_args.char_list[x] for x in nbest_hyps[0]['yseq']]
    text = [x for x in text if x not in ['<NOISE>', '<eos>']]
    text = ''.join(text).replace('<space>', ' ')

    logging.info(text)
            

def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")
    
    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    if args.backend == "chainer":
        recog(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    main(sys.argv[1:])

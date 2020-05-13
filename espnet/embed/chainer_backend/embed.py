#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import copy
import json
import logging
import os
import six

# chainer related
import chainer

from chainer.functions.loss import softmax_cross_entropy
from chainer import links as L
from chainer import functions as F
from chainer import link
from chainer import reporter
from chainer import training
from chainer import serializers

from chainer.datasets import TransformDataset
from chainer.training import extensions

# espnet related
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import CompareValue
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.scheduler.scheduler import dynamic_import_scheduler
from espnet.scheduler.chainer import ChainerScheduler
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

# numpy related
import matplotlib
import numpy as np

matplotlib.use('Agg')


class ClassifierWithoutState(link.Chain):
    """A wrapper for a chainer RNNLM

    :param link.Chain predictor : The RNNLM
    :param function lossfun: The loss function to use
    :param int/str label_key:
    """
    @staticmethod
    def add_arguments(parser):
        """Customize flags for transformer setup.

        Args:
            parser (Namespace): Training config.

        """
        group = parser.add_argument_group("X-vector model setting")
        group.add_argument("--layers", type=int, default=2,
                           help='')
        group.add_argument("--adim", type=int, default=256,
                           help='')
        group.add_argument("--dims", type=int, default=256,
                           help='')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='')
        group.add_argument('--feat-model', default=None, type=str,
                           help='')
        group.add_argument('--td-kernel', default=1, type=int,
                           help='')
        return parser

    def __init__(self, idim, odim, args):
        super(ClassifierWithoutState, self).__init__()
        self.y = None
        self.loss = None
        self.dropout = args.dropout_rate
        self.layers = args.layers
        self.kernel = args.td_kernel

        net = dynamic_import(args.feat_model)
        logging.info(net)
        channels = 64
        initial = chainer.initializers.Uniform
        dims = args.dims
        
        with self.init_scope():
            _idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
            self.feats = net(
                channels, _idim, args.adim, 0.1,
                initialW=initial, initial_bias=initial)
            _idim = args.adim
            self.td01 = L.Convolution1D(args.adim, dims, self.kernel)
            self.norm = L.GroupNormalization(1, dims)
            stvd = 1. / np.sqrt(dims)
            for i in range(args.layers):
                linear = L.Linear(dims, dims,
                                  initialW=initial(scale=stvd),
                                  initial_bias=initial(scale=stvd))
                setattr(self, f'linear{i}', linear)
            self.olinear = L.Linear(dims, odim,
                                    initialW=initial(scale=stvd),
                                    initial_bias=initial(scale=stvd))
        
        serializers.load_npz(args.enc_init, self.feats)

    def __call__(self, xs, ilens, ys):
        """Computes the loss value for an input and label pair.
        """
        xp = self.xp
        with chainer.no_backprop_mode():
            xs, ilens = self.feats(xs, ilens, no_pe=True)
        # logging.info(ilens)
        xs = F.relu(self.norm(self.td01(xs.transpose(0, 2, 1)))).transpose(0, 2, 1)
        ilens = (ilens - self.kernel) + 1
        xs = F.vstack([xs[i, :ilens[i]] for i in range(len(ilens))])
        for i in range(self.layers):
            xs = F.dropout(xs, self.dropout)
            xs = F.relu(self[f'linear{i}'](xs))
        xs = self.olinear(xs)
        ys = np.concatenate(
            [np.full((ilens[i]), ys[i], dtype=np.int32) for i in range(len(ilens))], axis=0)
        ys = xp.array(ys)
        self.loss = F.softmax_cross_entropy(xs, ys)
        self.acc = F.accuracy(xs, ys)
        return self.loss, self.acc
    
    def recognize(self, xs, args):
        ilens = [xs.shape[0]]
        xs, ilens = self.feats(xs[None], ilens, no_pe=True)
        # logging.info(ilens)
        xs = F.relu(self.norm(self.td01(xs.transpose(0, 2, 1)))).transpose(0, 2, 1)[0]
        for i in range(self.layers):
            xs = F.dropout(xs, self.dropout)
            xs = F.relu(self[f'linear{i}'](xs))
        xs = F.sum(F.log_softmax(self.olinear(xs)), axis=0).data
        xs = np.argsort(xs)[::-1][:5]
        return xs[0]


class CustomUpdater(training.updaters.StandardUpdater):
    """An updater for a chainer LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param optimizer:
    :param schedulers:
    :param int device : The device id
    :param int accum_grad :
    """

    def __init__(self, train_iter, optimizer, schedulers, device, converter):
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.scheduler = ChainerScheduler(schedulers, optimizer)
        self.converter = converter

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        optimizer.target.cleargrads()  # Clear the parameter gradients
        # Progress the dataset iterator for sentences at each iteration.
        batch = train_iter.__next__()
        x, ilens, t = self.converter(batch, device=self.device)
        loss, acc = optimizer.target(x, ilens, t)
        loss.backward()
        reporter.report({'loss': float(loss.data)}, optimizer.target)
        reporter.report({'acc': float(acc.data)}, optimizer.target)
        # update
        optimizer.update()  # Update the parameters
        # self.scheduler.step(self.iteration)


class CustomEvaluator(BaseEvaluator):
    """A custom evaluator for a chainer LM

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param eval_model : The model to evaluate
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, converter, device):
        super(CustomEvaluator, self).__init__(
            val_iter, eval_model, device=device)
        self.converter = converter

    def evaluate(self):
        val_iter = self.get_iterator('main')
        target = self.get_target('main')
        loss = list()
        acc = list()
        for batch in copy.copy(val_iter):
            x, ilens, t = self.converter(batch, device=self.device)
            _loss, _acc = target(x, ilens, t)
            loss += [float(_loss.data)]
            acc += [float(_acc.data)]
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'loss': np.average(loss)}, target)
            reporter.report({'acc': np.average(acc)}, target)
        return observation


class CustomConverter(object):
    """Custom Converter.

    Args:
        subsampling_factor (int): The subsampling factor.

    """

    def __init__(self, spks):
        """Initialize subsampling."""
        _spks = dict()
        for i, spk in enumerate(spks):
            _spks[str(spk)] = int(i)
        self.spks_dict = _spks

    def __call__(self, batch, device):
        """Perform subsampling.

        Args:
            batch (list): Batch that will be sabsampled.
            device (chainer.backend.Device): CPU or GPU device.

        Returns:
            chainer.Variable: xp.array that are padded and subsampled from batch.
            xp.array: xp.array of the length of the mini-batches.
            chainer.Variable: xp.array that are padded and subsampled from batch.

        """
        # For transformer, data is processed in CPU.
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]
        # get batch of lengths of input sequences
        ys = np.array([self.spks_dict.get(y, 0) for y in ys], dtype=np.int32)
        ilens = np.array([x.shape[0] for x in xs])
        xs = F.pad_sequence(xs, padding=-1).data
        return xs, ilens, ys


def add_json(gs, pd, spks_dict, spks_list):
    new_js = {
            'groundtruth': {
                'idx' : spks_dict.get(gs),
                'label' : gs },
            'prediction' : {
                'idx' : pd,
                'label' : spks_list[pd]
            }}
    return new_js


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    set_deterministic_chainer(args)

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning('cuda is not available')
    if not chainer.cuda.cudnn_enabled:
        logging.warning('cudnn is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = len(args.char_list)
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    model = ClassifierWithoutState(idim, odim, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    ngpu = args.ngpu
    if ngpu >= 1:
        gpu_id = 0
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
        logging.info('single gpu calculation.')
    else:
        gpu_id = -1
        logging.info('cpu calculation')

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta(eps=args.eps)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    elif args.opt == 'adabound':
        optimizer = chainer.optimizers.Adam(alpha=args.alpha, adabound=True)
    else:
        raise NotImplementedError('args.opt={}'.format(args.opt))

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # set up training iterator and updater
    load_tr = LoadInputsAndTargets(
        mode='embed', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='embed', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if args.schedulers is None:
        schedulers = []
    else:
        schedulers = [dynamic_import_scheduler(v)(k, args) for k, v in args.schedulers]

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                            args.maxlen_in, args.maxlen_out, args.minibatches,
                            min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                            shortest_first=use_sortagrad,
                            count=args.batch_count,
                            batch_bins=args.batch_bins,
                            batch_frames_in=args.batch_frames_in,
                            batch_frames_out=args.batch_frames_out,
                            batch_frames_inout=args.batch_frames_inout,
                            iaxis=0, oaxis=0)
    # hack to make batchsize argument as 1
    # actual batchsize is included in a list
    if args.n_iter_processes > 0:
        train_iters = [ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)]
    else:
        train_iters = [ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)]

    # set up updater
    converter=CustomConverter(args.char_list)
    updater = CustomUpdater(train_iters[0], optimizer, schedulers, gpu_id, converter)

    # Set up a trainer
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler(train_iters),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # set up validation iterator
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)

    if args.n_iter_processes > 0:
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(valid_iter, model, converter=converter,  device=gpu_id))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(
        filename='snapshot.ep.{.updater.epoch}', n_retains=10,
        condition=CompareValue(
            'validation/main/acc', lambda best_value, current_value: best_value < current_value, trainer)),
        trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_entropy', 'validation/main/loss_entropy',
                                          'main/loss_binary', 'validation/main/loss_binary'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, 'model.acc.best'),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))
    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best'),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best'),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').eps),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    elif args.opt in ['adam', 'adabound']:
        trainer.extend(extensions.observe_value(
            'alpha', lambda trainer: trainer.updater.get_optimizer('main').alpha),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('alpha')

    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))
    
    if 'dump_plot' in args:
        if args.dump_plot:
            trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))

    set_early_stop(trainer, args)
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


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
    chainer_load(args.model, model)

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='embed', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    _spks = dict()
    for i, spk in enumerate(train_args.char_list):
        _spks[str(spk)] = int(i)
    spks_dict = _spks
    # decode each utterance
    new_js = {}
    idx_gt = list()
    idx_pd = list()
    with chainer.no_backprop_mode():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            nbest_hyps = model.recognize(feat, args)

            gs = js[name]['utt2spk']
            idx_gt.append(spks_dict.get(gs))
            idx_pd.append(nbest_hyps)
            logging.info(f'groundtruth: {gs}, prediction: {train_args.char_list[nbest_hyps]}')
            new_js[name] = add_json(gs, int(nbest_hyps), spks_dict, train_args.char_list)
    acc = np.mean(np.array(idx_gt) == np.array(idx_pd)) * 100
    logging.info(f'SID: {acc:.02f} %')

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

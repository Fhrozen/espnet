#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import math
import sys

from argparse import Namespace

import numpy as np
import random
import six

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer import reporter
from chainer_ctc.warpctc import ctc as warp_ctc
from ctc_prefix_score import CTCPrefixScore
from e2e_asr_common import end_detect
from e2e_asr_common import get_vgg2l_odim
from e2e_asr_common import label_smoothing_dist
import kaldi_io_py
from net_utils import GridLSTMCell

import deterministic_embed_id as DL

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


def _subsamplex(x, n):
    x = [F.get_item(xx, (slice(None, None, n), slice(None))) for xx in x]
    ilens = [xx.shape[0] for xx in x]
    return x, ilens


# TODO(kan-bayashi): no need to use linear tensor
def linear_tensor(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable y: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    y = linear(F.reshape(x, (-1, x.shape[-1])))
    return F.reshape(y, (x.shape[:-1] + (-1,)))


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(chainer.Chain):
    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, xs, ilens, ys):
        '''Loss forward

        :param x:
        :return:
        '''
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(xs, ilens, ys)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
        elif alpha == 1:
            self.loss = loss_ctc
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att

        if self.loss.data < CTC_LOSS_THRESHOLD and not math.isnan(self.loss.data):
            reporter.report({'loss_ctc': loss_ctc}, self)
            reporter.report({'loss_att': loss_att}, self)
            reporter.report({'acc': acc}, self)

            logging.info('mtl loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)

        return self.loss


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        with self.init_scope():
            # encoder
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate, args.einputs, args.norm_file)
            # ctc
            ctc_type = vars(args).get("ctc_type", "chainer")
            if ctc_type == 'chainer':
                logging.info("Using chainer CTC implementation")
                self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
            elif ctc_type == 'warpctc':
                logging.info("Using warpctc CTC implementation")
                self.ctc = WarpCTC(odim, args.eprojs, args.dropout_rate)
            # attention
            if args.atype == 'dot':
                self.att = AttDot(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'location':
                self.att = AttLoc(args.eprojs, args.dunits,
                                  args.adim, args.aconv_chans, args.aconv_filts)
            elif args.atype == 'add':
                self.att = AttAdd(args.eprojs, args.dunits, args.adim)
            elif args.atype == 'noatt':
                self.att = NoAtt()
            else:
                logging.error(
                    "Error: need to specify an appropriate attention archtecture")
                sys.exit()
            # decoder
            self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                               self.sos, self.eos, self.att, self.verbose, self.char_list,
                               labeldist, args.lsm_weight, args.sampling_probability)

    def __call__(self, xs, ilens, ys):
        '''E2E forward

        :param data:
        :return:
        '''
        # 1. encoder
        hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs, ys)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E greedy/beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = self.xp.array(x.shape[0], dtype=np.int32)
        h = chainer.Variable(self.xp.array(x, dtype=np.float32))

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # 1. encoder
            # make a utt list (1) to use the same interface for encoder
            h, _ = self.enc([h], [ilen])
            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(h).data[0]
            else:
                lpz = None

            # 2. decoder
            # decode the first utterance
            y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

        return y

    def calculate_all_attentions(self, xs, ilens, ys):
        '''E2E attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws


# ------------- CTC Network --------------------------------------------------------------------------------------------
class CTC(chainer.Chain):
    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        '''CTC forward

        :param hs:
        :param ys:
        :return:
        '''
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = linear_tensor(self.ctc_lo, F.dropout(
            F.pad_sequence(hs), ratio=self.dropout_rate))
        y_hat = F.separate(y_hat, axis=1)  # ilen list of batch x hdim

        # zero padding for ys
        y_true = F.pad_sequence(ys, padding=-1)  # batch x olen

        # get length info
        input_length = chainer.Variable(self.xp.array(ilens, dtype=np.int32))
        label_length = chainer.Variable(self.xp.array(olens, dtype=np.int32))
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(input_length.data))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(label_length.data))

        # get ctc loss
        self.loss_nomean = F.connectionist_temporal_classification(
            y_hat, y_true, 0, input_length, label_length, reduce='no')
        mask = [0 if l > CTC_LOSS_THRESHOLD else 1 for l in self.loss_nomean.data]
        masked_loss = [a*b for a, b in zip(mask, self.loss_nomean)]
        if sum(mask) == 0:
            self.loss = F.mean(self.loss_nomean) * 0
        else:
            self.loss = sum(masked_loss) / sum(mask)
        logging.info('ctc loss:' + str(self.loss.data))

        return self.loss

    def log_softmax(self, hs):
        '''log_softmax of frame activations

        :param hs:
        :return:
        '''
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)


class WarpCTC(chainer.Chain):
    def __init__(self, odim, eprojs, dropout_rate):
        super(WarpCTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None

        with self.init_scope():
            self.ctc_lo = L.Linear(eprojs, odim)

    def __call__(self, hs, ys):
        '''CTC forward

        :param hs:
        :param ys:
        :return:
        '''
        self.loss = None
        ilens = [x.shape[0] for x in hs]
        olens = [x.shape[0] for x in ys]

        # zero padding for hs
        y_hat = linear_tensor(self.ctc_lo, F.dropout(
            F.pad_sequence(hs), ratio=self.dropout_rate))
        y_hat = F.transpose(y_hat, (1, 0, 2))  # batch x frames x hdim

        # get length info
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(ilens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(olens))

        # get ctc loss
        self.loss = warp_ctc(y_hat, ilens, [cuda.to_cpu(l.data) for l in ys])[0]
        logging.info('ctc loss:' + str(self.loss.data))

        return self.loss

    def log_softmax(self, hs):
        '''log_softmax of frame activations

        :param hs:
        :return:
        '''
        y_hat = linear_tensor(self.ctc_lo, F.pad_sequence(hs))
        return F.log_softmax(y_hat.reshape(-1, y_hat.shape[-1])).reshape(y_hat.shape)


# ------------- Attention Network --------------------------------------------------------------------------------------
# dot product based attention
class AttDot(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param enc_hs:
        :param dec_z:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(
                linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # <phi (h_t), psi (s)> for all t
        u = F.broadcast_to(F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1),
                           self.pre_compute_enc_h.shape)
        e = F.sum(self.pre_compute_enc_h * u, axis=2)  # utt x frame
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)
        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


# location based attention
class AttLoc(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim, nobias=True)
            self.mlp_att = L.Linear(aconv_chans, att_dim, nobias=True)
            self.loc_conv = L.Convolution2D(1, aconv_chans, ksize=(
                1, 2 * aconv_filts + 1), pad=(0, aconv_filts))
            self.gvec = L.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param enc_hs:
        :param dec_z:
        :param att_prev:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # TODO(watanabe) use <chainer variable>.reshpae(), instead of F.reshape()
        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(
            F.reshape(att_prev, (batch, 1, 1, self.h_length)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO(watanabe) use batch_matmul
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


class AttAdd(chainer.Chain):
    '''Additive attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttAdd, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim, nobias=True)
            self.gvec = L.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def __call__(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param Variable enc_hs: padded encoder hidden state (B x T_max x D_enc)
        :param Variable dec_z: decoder hidden state (B x D_dec)
        :param Variable att_prev: dummy
        :param float scaling: scaling parameter before applying softmax
        :return: ``(c, w)``, where ``c`` represents attention weighted encoder
         state (B, D_enc), and ``w`` is previous attention weights (B x T_max)
        :rtype: tuple of (~chainer.Variable)
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # NOTE consider zero padding when compute w.
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(
            self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


class NoAtt(chainer.Chain):
    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def __call__(self, enc_hs, dec_z, att_prev):
        '''NoAtt forward

        :param enc_hs:
        :param dec_z: dummy
        :param att_prev:
        :return:
        '''
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)
            self.c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(att_prev, 2), self.enc_h.shape), axis=1)

        return self.c, att_prev


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(chainer.Chain):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(odim, dunits)
            self.lstm0 = L.StatelessLSTM(dunits + eprojs, dunits)
            for l in six.moves.range(1, dlayers):
                setattr(self, 'lstm%d' % l, L.StatelessLSTM(dunits, dunits))
            self.output = L.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability

    def __call__(self, hs, ys):
        '''Decoder forward

        :param Variable hs:
        :param Variable ys:
        :return:
        '''
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get dim, length info
        batch = pad_ys_out.shape[0]
        olength = pad_ys_out.shape[1]
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(self.xp.array([h.shape[0] for h in hs])))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(self.xp.array([y.shape[0] for y in ys_out])))

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for l in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = F.argmax(F.log_softmax(z_out), axis=1)
                z_out = self.embed(z_out)
                ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            else:
                ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            z_all.append(z_list[-1])

        z_all = F.reshape(F.stack(z_all, axis=1),
                          (batch * olength, self.dunits))
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        logging.info('att loss:' + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = F.reshape(y_all, (batch, olength, -1))
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat).replace('<space>', ' ')
                seq_true = "".join(seq_true).replace('<space>', ' ')
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
            loss_reg = - F.sum(F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param h:
        :param recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.shape[0]))
        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for l in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.xp.full(1, self.sos, 'i')
        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.shape[0]))
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                ey = self.embed(hyp['yseq'][i])           # utt list (1) x zdim
                att_c, att_w = self.att([h], hyp['z_prev'][0], hyp['a_prev'])
                ey = F.hstack((ey, att_c))   # utt(1) x (zdim + hdim)
                c_list[0], z_list[0] = self.lstm0(hyp['c_prev'][0], hyp['z_prev'][0], ey)
                for l in six.moves.range(1, self.dlayers):
                    c_list[l], z_list[l] = self['lstm%d' % l](
                        hyp['c_prev'][l], hyp['z_prev'][l], z_list[l - 1])

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1])).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], hyp['yseq'][i])
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:ctc_beam]
                    ctc_scores, ctc_states = ctc_prefix_score(hyp['yseq'], local_best_ids, hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids] \
                        + ctc_weight * (ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids]
                    joint_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, joint_best_ids]
                    local_best_ids = local_best_ids[joint_best_ids]
                else:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, local_best_ids]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # do not copy {z,c}_list directly
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = self.xp.full(
                        1, local_best_ids[j], 'i')
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug('best hypo: ' + ''.join([char_list[int(x)]
                                                   for x in hyps[0]['yseq'][1:]]).replace('<space>', ' '))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.xp.full(1, self.eos, 'i'))

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug('hypo: ' + ''.join([char_list[int(x)]
                                                  for x in hyp['yseq'][1:]]).replace('<space>', ' '))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        return nbest_hyps

    def calculate_all_attentions(self, hs, ys):
        '''Calculate all of attentions

        :return: list of attentions
        '''
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get length info
        olength = pad_ys_out.shape[1]

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for l in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            att_ws.append(att_w)  # for debugging

        att_ws = F.stack(att_ws, axis=1)
        att_ws.to_cpu()

        return att_ws.data


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(chainer.Chain):
    '''ENCODER NETWORK CLASS

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param str subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

    '''

    def __init__(self, etype, idim, elayers, eunits,
                 eprojs, subsample, dropout, in_channel=1, normfile=None):
        super(Encoder, self).__init__()
        encoders = etype.split('_')
        nopad = False
        with self.init_scope():
            self._forward = list()
            for i in range(len(encoders)):
                name = 'enc{}'.format(i + 1)
                _e = encoders[i].split('.')
                e_type = _e[0]
                e_spec = {'m': 'single',
                          'b': None,
                          'o': 128,
                          'd': None,
                          'r': 0.0,
                          's': '2222',
                          'a': F.relu,
                          'p': 0,
                          'l': 2}

                for j in range(len(_e) - 1):
                    prefix = _e[j + 1][0]
                    if prefix == 'o' or prefix == 'p' or prefix == 'l':
                        val = int(_e[j + 1][1:])
                    elif prefix == 'r':
                        val = float(_e[j + 1][1:]) / 10.
                    elif prefix == 'a':
                        val = getattr(F, str(_e[j + 1][1:]))
                    else:
                        val = str(_e[j + 1][1:])
                    e_spec[prefix] = val

                if e_type == 'blstm':
                    _encoder = BLSTM(idim, elayers, eunits, eprojs, dropout)
                    logging.info('BLSTM added for encoder')
                elif e_type == 'blstmp':
                    _encoder = BLSTMP(idim, elayers, eunits,
                                      eprojs, subsample, dropout)
                    logging.info('BLSTM with every-layer projection added for encoder')
                elif e_type == 'lstmp':
                    _encoder = LSTMP(idim, elayers, eunits,
                                      eprojs, subsample, dropout)
                    logging.info('LSTM with every-layer projection added for encoder')
                elif e_type == 'bgrup':
                    _encoder = BGRUP(idim, elayers, eunits,
                                     eprojs, subsample, dropout)
                    logging.info('BiGRU with every-layer projection added for encoder')
                elif e_type == 'lstm':
                    _encoder = LSTM(idim, elayers, eunits,
                                    eprojs, subsample, dropout)
                    logging.info('LSTM added for encoder')
                elif e_type == 'mcblstmp':
                    _encoder = MultiChannelBLSTMP(in_channel, idim, elayers, eunits,
                                                  eprojs, subsample, dropout)
                    logging.info('BLSTM with every-layer projection added for encoder')
                elif e_type == 'grid':
                    _encoder = GridLSTM(idim, elayers, eunits,
                                        eprojs, subsample, dropout)
                    logging.info('GridLSTM added for encoder')
                elif e_type == 'vgg':
                    _encoder = VGG2L(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                     nopad=nopad, reshape=e_spec['p'])
                    idim = get_vgg2l_odim(idim, subsample=e_spec['s'])
                    logging.info('CNN-VGG with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'res':
                    _encoder = RESNET(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                      dropout=e_spec['d'], dratio=e_spec['r'], act=e_spec['a'],
                                      bn=e_spec['b'], outs=e_spec['o'], nopad=nopad, reshape=e_spec['p'])
                    idim = get_vgg2l_odim(idim, out_channel=e_spec['o'], subsample=e_spec['s'])
                    logging.info('CNN-RESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'dense':
                    _encoder = DENSENET(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                      dropout=e_spec['d'], dratio=e_spec['r'], act=e_spec['a'],
                                      bn=e_spec['b'], outs=e_spec['o'], nopad=nopad, reshape=e_spec['p'])
                    idim = get_vgg2l_odim(idim, out_channel=e_spec['o'], subsample=e_spec['s'])
                    logging.info('CNN-RESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'resspk':
                    _encoder = RESSPK(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                      dropout=e_spec['d'], dratio=e_spec['r'], act=e_spec['a'],
                                      bn=e_spec['b'], outs=e_spec['o'], nopad=nopad, reshape=e_spec['p'])
                    idim = get_vgg2l_odim(idim, out_channel=e_spec['o']+1, subsample=e_spec['s'])
                    logging.info('CNN-RESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'tres':
                    _encoder = TRESNET(e_spec['o'], frames=e_spec['l'], bn=e_spec['b'], projs=eunits)
                    logging.info('CNN-TRESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'tlmres':
                    _encoder = TLMRESNET(e_spec['o'], frames=e_spec['l'], bn=e_spec['b'])
                    logging.info('CNN-TRESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'lmres':
                    _encoder = LMRESNET(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                        dropout=e_spec['d'], dratio=e_spec['r'], act=e_spec['a'],
                                        bn=e_spec['b'], outs=e_spec['o'], nopad=nopad, reshape=e_spec['p'])
                    idim = get_vgg2l_odim(idim, out_channel=e_spec['o'], subsample=e_spec['s'])
                    logging.info('CNN-LM-RESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'resloc':
                    _encoder = RESLOC(in_channel, mode=e_spec['m'], subsample=e_spec['s'],
                                        dropout=e_spec['d'], dratio=e_spec['r'], act=e_spec['a'],
                                        bn=e_spec['b'], outs=e_spec['o'], nopad=nopad)
                    idim = get_vgg2l_odim(idim, out_channel=e_spec['o'], subsample=e_spec['s'])
                    logging.info('CNN-LM-RESNET with specs {} added for encoder'.format(_e[1:]))
                elif e_type == 'fn':
                    _encoder = filternet(3, nchannels=in_channel)
                    in_channel = 1
                    logging.info('FilterNet added for encoder')
                    idim = 40
                    nopad = True
                elif e_type == 'ftprev':
                    _encoder = LMFILTPREV(in_channel)
                    logging.info('Residual Filter with previous frame added for encoder')
                    nopad = True
                elif e_type == 'ftlat':
                    _encoder = LMFILTLT(in_channel)
                    logging.info('LM-Res Filter with lateral frames added for encoder')
                    nopad = True
                else:
                    logging.error(
                        "Error: {} not found. Need to specify an appropriate encoder archtecture".format(e_type))
                    sys.exit(1)
                setattr(self, name, _encoder)
                self._forward.append(name)
        self.etype = etype

    def __call__(self, xs, ilens):
        '''Encoder forward

        :param xs:
        :param ilens:
        :return:
        '''
        for name in self._forward:
            enc = getattr(self, name)
            xs, ilens = enc(xs, ilens)
        return xs, ilens

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


# TODO(watanabe) explanation of BLSTMP
class MultiChannelBLSTMP(chainer.Chain):
    def __init__(self, in_channels, idim, elayers, cdim, hdim, subsample, dropout):
        super(MultiChannelBLSTMP, self).__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        comb = [(x, y) for x in six.moves.range(len(in_channels)) for y in six.moves.range(elayers)]
        with self.init_scope():
            for i, j in comb:
                if j == 0:
                    inputdim = idim * in_channels[i]
                else:
                    inputdim = hdim
                setattr(self, 'bilstm{}_l{}'.format(i, j), L.NStepBiLSTM(
                    1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                setattr(self, 'bt{}_l{}'.format(i, j), L.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.in_channels = in_channels
        self.idim = idim

    def __call__(self, xs, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        # xs: tuple[frames x channels x dims]
        bs = len(xs)
        ch = xs[0].shape[1]
        idx = self.in_channels.index(ch)
        xs = [F.reshape(xs[i], (int(ilens[i]), ch * self.idim)) for i in six.moves.range(bs)]
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        for layer in six.moves.range(self.elayers):
            hy, cy, ys = self['bilstm{}_l{}'.format(idx, layer)](None, None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt{}_l{}'.format(idx, layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


# TODO(watanabe) explanation of BLSTMP
class BLSTMP(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                setattr(self, "bilstm%d" % i, L.NStepBiLSTM(
                    1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                setattr(self, "bt%d" % i, L.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def __call__(self, xs, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            hy, cy, ys = self['bilstm' + str(layer)](None, None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt' + str(layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class LSTMP(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(LSTMP, self).__init__()
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                setattr(self, "lstm%d" % i, L.NStepLSTM(
                    1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                setattr(self, "bt%d" % i, L.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def __call__(self, xs, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            hy, cy, ys = self['lstm' + str(layer)](None, None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt' + str(layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class BGRUP(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BGRUP, self).__init__()
        with self.init_scope():
            for i in six.moves.range(elayers):
                if i == 0:
                    inputdim = idim
                else:
                    inputdim = hdim
                setattr(self, "bigru%d" % i, L.NStepBiGRU(
                    1, inputdim, cdim, dropout))
                # bottleneck layer to merge
                setattr(self, "bt%d" % i, L.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def __call__(self, xs, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            hy, ys = self['bigru' + str(layer)](None, xs)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # TODO(watanabe) replace subsample and FC layer with CNN
            ys, ilens = _subsamplex(ys, self.subsample[layer + 1])
            # (sum _utt frame_utt) x dim
            ys = self['bt' + str(layer)](F.vstack(ys))
            xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class GridLSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(GridLSTM, self).__init__()
        shared_dims = [[1], [x for x in range(2, elayers + 1)]]

        with self.init_scope():
            self.grid = GridLSTMCell([idim, cdim], shared_dims, elayers)
            self.l_last = L.Linear(cdim * 2, hdim)

    def __call__(self, ys, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        ilens = np.asarray([x for x in ilens], dtype=np.int32)
        cy = None
        for x in range(ys.shape[1]):
            cy, ys = self.grid(cy, ys[x])
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class LSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.nblstm = L.NStepBiLSTM(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(cdim * 2, hdim)

    def __call__(self, xs, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        ilens = np.asarray([x for x in ilens], dtype=np.int32)
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class BLSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        with self.init_scope():
            self.nblstm = L.NStepBiLSTM(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(cdim * 2, hdim)

    def __call__(self, xs, ilens):
        '''BLSTM forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # need to move ilens to cpu
        ilens = cuda.to_cpu(ilens)
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


# TODO(watanabe) explanation of VGG2L, VGG2B (Block) might be better
class VGG2L(chainer.Chain):
    def __init__(self, in_channel=1, mode='single', subsample='2222', nopad=False, reshape=0):
        super(VGG2L, self).__init__()
        if isinstance(in_channel, int):
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(4)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(4)]
        elif mode == 'entry':
            combs = [(x, 0) for x in range(len(in_channel))] + [(0, y) for y in range(1, 4)]
        else:
            raise ValueError('Incorrect mode.')

        with self.init_scope():
            # CNN layer (VGG motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 64, 3, stride=1, pad=1)
                elif j == 1:
                    layer = L.Convolution2D(64, 64, 3, stride=1, pad=1)
                elif j == 2:
                    layer = L.Convolution2D(64, 128, 3, stride=1, pad=1)
                else:
                    layer = L.Convolution2D(128, 128, 3, stride=1, pad=1)
                setattr(self, l_name, layer)
        self.in_channel = in_channel
        self.mode = mode
        self.nopad = nopad
        self.reshape = reshape
        if subsample == '2222':
            self.subsample = self.subsample2222
        elif subsample == '3111':
            self.subsample = self.subsample3111
        else:
            raise ValueError('Incorrect type of subsample')

    def __call__(self, xs, ilens):
        '''VGG2L forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # x: utt x frame x input channel x dim
        if not self.nopad:
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)
        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        xs = F.relu(self['conv{}_0'.format(idx)](xs))
        if self.mode == 'entry':
            idx = 0
        xs = F.relu(self['conv{}_1'.format(idx)](xs))
        xs, ilens = self.subsample(idx, xs, ilens)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        if self.reshape:
            xs = F.swapaxes(xs, 1, 2)
            xs = F.reshape(
                xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
            xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]
        return xs, ilens

    def subsample2222(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = F.relu(self['conv{}_2'.format(idx)](xs))
        xs = F.relu(self['conv{}_3'.format(idx)](xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        return xs, ilens

    def subsample3111(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, (3, 1), stride=(3, 1))

        xs = F.relu(self['conv{}_2'.format(idx)](xs))
        xs = F.relu(self['conv{}_3'.format(idx)](xs))

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 3), dtype=np.int32)
        return xs, ilens


class ConvWithBReNorm(chainer.Chain):
    def __init__(self, in_channels, out_channels,
                 kernel=1, stride=1, pad=0, nobias=True, groups=1):
        super(ConvWithBReNorm, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, out_channels, kernel, stride=stride, pad=pad, nobias=nobias, groups=groups)
            self.bn = L.BatchRenormalization(out_channels)

    def __call__(self, x):
        x = self.conv(x)
        return self.bn(x)


class BottleneckA(chainer.Chain):
    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, bn=None, act=F.relu, groups=1):
        super(BottleneckA, self).__init__()
        if bn == 'ReNorm':
            Conv = ConvWithBReNorm
        else:
            Conv = L.Convolution2D
        with self.init_scope():
            self.shortcut = Conv(in_channels, out_channels, 1, stride=stride, pad=0, nobias=True, groups=groups)
            self.conv1 = Conv(in_channels, mid_channels, 3, stride=1, pad=1, nobias=True, groups=groups)
            self.conv2 = Conv(mid_channels, out_channels, 3, stride=stride, pad=1, nobias=True, groups=groups)
        self.act = act
        self.bn = bn

    def __call__(self, x):
        res_x = self.act(self.conv1(x))
        res_x = self.conv2(res_x)
        x = self.shortcut(x)
        return self.act(x + res_x)


class LMBottleneckA(chainer.Chain):
    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, bn=None, act=F.relu, k=0.5):
        super(LMBottleneckA, self).__init__()
        if bn == 'ReNorm':
            Conv = ConvWithBReNorm
        else:
            Conv = L.Convolution2D
        with self.init_scope():
            self.k = chainer.variable.Parameter(0.5, ())
            self.shortcut = Conv(in_channels, out_channels, 1, stride=stride, pad=0, nobias=True)
            self.conv1_1 = Conv(in_channels, mid_channels, 3, stride=1, pad=1, nobias=True)
            self.conv1_2 = Conv(mid_channels, out_channels, 3, stride=stride, pad=1, nobias=True)
            self.conv2_1 = Conv(out_channels, mid_channels, 3, stride=1, pad=1, nobias=True)
            self.conv2_2 = Conv(mid_channels, out_channels, 3, stride=1, pad=1, nobias=True)
        self.bn = bn
        self.act = act

    def __call__(self, x0):
        # Output: Un+1 = (1 - kn)*Un + kn*Un-1 + f(Un)
        # Un = act(Un-1 + f(Un-1))
        f_x0 = self.act(self.conv1_1(x0))
        f_x0 = self.conv1_2(f_x0)
        x0 = self.shortcut(x0)
        x1 = self.act(x0 + f_x0)
        dims = x1.shape
        f_x1 = self.act(self.conv2_1(x1))
        f_x1 = self.conv2_2(f_x1)
        x1 = (1. - F.broadcast_to(self.k, dims)) * x1
        x2 = x1 + F.broadcast_to(self.k, dims) * x0 + f_x1
        return self.act(x2)


class PreBottleneckA(chainer.Chain):
    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, initialW=None, act=F.relu):
        super(PreBottleneckA, self).__init__()
        with self.init_scope():
            self.shortcut = L.Convolution2D(
                in_channels, out_channels, 1, stride=stride, pad=0, nobias=True)
            self.bn1 = L.BatchRenormalization(in_channels)
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 3, stride=1, pad=1, nobias=True)
            self.conv2 = L.Convolution2D(
                mid_channels, out_channels, 3, stride=stride, pad=1, nobias=True)
            self.bn2 = L.BatchRenormalization(mid_channels)
        self.act = act

    def __call__(self, x):
        res_x = self.conv1(self.act(self.bn1(x)))
        res_x = self.conv2(self.act(self.bn2(res_x)))
        x = self.shortcut(x)
        return x + res_x


class BottleneckB(chainer.Chain):
    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 3, stride=1, pad=1, nobias=True)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, stride=1, pad=1, nobias=True)

    def __call__(self, x):
        res_x = F.relu(self.conv1(x))
        res_x = self.conv2(res_x)
        return F.relu(x + res_x)


class BuildingBlock(chainer.Chain):
    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            layer = getattr(self, name)
            x = layer(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


def dropout_fixed(xs, ratio, _iter):
    return F.dropout(xs, ratio)


def dropout_incremental(xs, ratio, _iter):
    # delayed approx 2 epocs
    ratio = min(max(_iter - 20000, 0) / 100000.0, 0.99)
    if ratio > 0.0:
        return F.dropout(xs, ratio)
    return xs


def dropout_random(xs, ratio, _iter):
    # delayed approx 2 epocs
    if max(_iter - 20000, 0.0) > 0.0:
        ratio = np.random.rand(1)[0]
        return F.dropout(xs, ratio)
    return xs


class RESNET(chainer.Chain):
    def __init__(self, in_channel=1, mode=None, act=F.relu, bn=None,
                 outs=128, dropout=None, dratio=0.0, nopad=False, subsample='2222', reshape=0):
        super(RESNET, self).__init__()
        if type(in_channel) is int:
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(3)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(3)]
        elif mode == 'entry':
            combs = [(x, 0) for x in range(len(in_channel))] + [(0, y) for y in range(1, 3)]
        else:
            raise ValueError('Incorrect mode.')
        with self.init_scope():
            # CNN layer (RESNET motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 16, 1, stride=1, nobias=True)
                elif j == 1:
                    layer = BottleneckA(16, 64, 64, act=act, bn=bn)
                else:
                    layer = BottleneckA(64, 128, outs, act=act, bn=bn)
                setattr(self, l_name, layer)

        for x in range(len(in_channel)):
            doutname = 'drop_{}'.format(x)
            douttype = None
            if in_channel[x] == 2:
                if dropout == 'inc':
                    logging.info('Adding Incremental dropout to the training')
                    douttype = dropout_incremental
                elif dropout == 'fix':
                    logging.info('Adding fixed dropout to the training')
                    douttype = dropout_fixed
                elif dropout == 'rand':
                    logging.info('Adding random dropout to the training')
                    douttype = dropout_random
            setattr(self, doutname, douttype)

        self.dropout = dropout
        self.in_channel = in_channel
        self.mode = mode
        self.iter = 0
        self.dratio = dratio
        self.nopad = nopad
        self.reshape = reshape
        if subsample == '2222':
            self.subsample = self.subsample2222
        elif subsample == '3111':
            self.subsample = self.subsample3111
        else:
            raise ValueError('Incorrect type of subsample')

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        self.iter += 1
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        if not self.nopad:
            # x: utt x frame x input channel x dim
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)

        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        # Apply dropout only to the binaural input
        if not self['drop_{}'.format(idx)] is None:
            xs = self['drop_{}'.format(idx)](xs, self.dratio, self.iter)

        xs = self['conv{}_0'.format(idx)](xs)
        if self.mode == 'entry':
            idx = 0

        xs = self['conv{}_1'.format(idx)](xs)
        xs, ilens = self.subsample(idx, xs, ilens)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)+
        if self.reshape:
            xs = F.swapaxes(xs, 1, 2)
            xs = F.reshape(
                xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
            xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens

    def subsample2222(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = self['conv{}_2'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        return xs, ilens

    def subsample3111(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, (3, 1), stride=(3, 1))

        xs = self['conv{}_2'.format(idx)](xs)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 3), dtype=np.int32)
        return xs, ilens


class RESSPK(chainer.Chain):
    def __init__(self, in_channel=1, mode=None, act=F.relu, bn=None,
                 outs=128, dropout=None, dratio=0.0, nopad=False, subsample='2222', reshape=0):
        super(RESSPK, self).__init__()
        if type(in_channel) is int:
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(3)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(3)]
        elif mode == 'entry':
            combs = [(x, 0) for x in range(len(in_channel))] + [(0, y) for y in range(1, 3)]
        else:
            raise ValueError('Incorrect mode.')
        with self.init_scope():
            # CNN layer (RESNET motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 16, 1, stride=1, nobias=True)
                elif j == 1:
                    layer = BottleneckA(16, 64, 64, act=act, bn=bn)
                else:
                    layer = BottleneckA(64, 128, outs, act=act, bn=bn)
                setattr(self, l_name, layer)
            self.spk = L.Convolution2D(128, 1, 3, stride=1)

        for x in range(len(in_channel)):
            doutname = 'drop_{}'.format(x)
            douttype = None
            if in_channel[x] == 2:
                if dropout == 'inc':
                    logging.info('Adding Incremental dropout to the training')
                    douttype = dropout_incremental
                elif dropout == 'fix':
                    logging.info('Adding fixed dropout to the training')
                    douttype = dropout_fixed
                elif dropout == 'rand':
                    logging.info('Adding random dropout to the training')
                    douttype = dropout_random
            setattr(self, doutname, douttype)

        self.dropout = dropout
        self.in_channel = in_channel
        self.mode = mode
        self.iter = 0
        self.dratio = dratio
        self.nopad = nopad
        self.reshape = reshape

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        self.iter += 1
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        if not self.nopad:
            # x: utt x frame x input channel x dim
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)

        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        # Apply dropout only to the binaural input
        if not self['drop_{}'.format(idx)] is None:
            xs = self['drop_{}'.format(idx)](xs, self.dratio, self.iter)

        xs = self['conv{}_0'.format(idx)](xs)
        if self.mode == 'entry':
            idx = 0

        xs = self['conv{}_1'.format(idx)](xs)
        xs, ilens = self.subsample(idx, xs, ilens)

        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = self['conv{}_2'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)
        xs_shape = xs.shape
        xs_shape[1] = 1
        spk = F.average(F.sum(F.relu(self.spk(xs)), axis=2, keepdims=True), axis=3, keepdims=True)
        spk = F.broadcast_to(spk, xs_shape)
        xs = F.concat((xs, spk), axis=1)
        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)+
        
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens


class TRESNET(chainer.Chain):
    def __init__(self, in_channel=128, frames=2, bn=None, projs=512):
        super(TRESNET, self).__init__()
        if bn == 'ReNorm':
            Conv = ConvWithBReNorm
        else:
            Conv = L.Convolution2D
        with self.init_scope():
            self.conv0 = Conv(in_channel, in_channel, (frames, 1), stride=1, nobias=True)
            self.out = L.Linear(projs)
        self.frames = frames

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # x: utt x input channel x frame x dim
        # logging.info(xs.shape)
        # xs = F.sum(xs, axis=1, keepdims=True)
        # logging.info(xs.shape)
        ilens = ilens - self.frames + 1
        length = xs.shape[2] - self.frames + 1
        xs_out = xs[:, :, 0:length, :]
        xs = self.conv0(xs)
        # logging.info(xs_out.shape)
        # logging.info(xs.shape)
        xs = F.relu(xs_out + xs)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        # Final Step Projection from ndims to odims
        xs = F.relu(self.out(F.vstack(xs)))
        xs = F.split_axis(xs, np.cumsum(self.xp.asnumpy(ilens[:-1])), axis=0)
        return xs, ilens


class LMRESNET(chainer.Chain):
    def __init__(self, in_channel=1, mode=None, act=F.relu, bn=None,
                 outs=128, dropout=None, dratio=0.0, nopad=False, subsample='2222'):
        super(LMRESNET, self).__init__()
        if type(in_channel) is int:
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(2)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(2)]
        elif mode == 'entry':
            combs = [(x, 0) for x in range(len(in_channel))] + [(0, y) for y in range(1, 2)]
        else:
            raise ValueError('Incorrect mode.')
        with self.init_scope():
            # CNN layer (RESNET motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 16, 1, stride=1, nobias=True)
                else:
                    layer = LMBottleneckA(16, 64, outs, stride=2, act=act, bn=bn)
                setattr(self, l_name, layer)

        for x in range(len(in_channel)):
            doutname = 'drop_{}'.format(x)
            douttype = no_dropout
            if in_channel[x] == 2:
                if dropout == 'inc':
                    logging.info('Adding Incremental dropout to the training')
                    douttype = dropout_incremental
                elif dropout == 'fix':
                    logging.info('Adding fixed dropout to the training')
                    douttype = dropout_fixed
                elif dropout == 'rand':
                    logging.info('Adding random dropout to the training')
                    douttype = dropout_random
            setattr(self, doutname, douttype)

        self.dropout = dropout
        self.in_channel = in_channel
        self.mode = mode
        self.iter = 0
        self.nopad = nopad
        self.dratio = dratio

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        self.iter += 1
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        if not self.nopad:
            # x: utt x frame x input channel x dim
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)
        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        # Apply dropout only to the binaural input
        xs = self['drop_{}'.format(idx)](xs, self.dratio, self.iter)

        xs = self['conv{}_0'.format(idx)](xs)
        if self.mode == 'entry':
            idx = 0

        xs = self['conv{}_1'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1], dtype=np.float32)
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def get_norm(filename):
    mat = kaldi_io_py.read_mat(filename)
    logging.info('Norm File size {}'.format(mat.shape))
    count = mat[0, 80]
    logging.info('Counted Files {}'.format(count))
    mean = mat[1, :80] / count
    var = (mat[0, :80] / count) - mean * mean
    floor = 1e-20
    var[var < floor] = floor
    scale = 1. / np.sqrt(var)
    offset = -1. * (mean * scale)
    return offset, scale


class filternet(chainer.Chain):
    def __init__(self, layers=1, nchannels=1, hdim=320, cdim=320, nfft=512, nfilt=40):
        super(filternet, self).__init__()
        if isinstance(nchannels, int):
            nchannels = [nchannels]
        combs1 = [(x, y) for x in range(len(nchannels)) for y in range(layers)]
        combs2 = [(x, y) for x in range(len(nchannels)) for y in range(2)]
        with self.init_scope():
            for i, j in combs1:
                if j == 0:
                    inputdim = nchannels[i] * 2 * (nfft // 2 + 1)
                else:
                    inputdim = hdim
                setattr(self, "bilstm{}_l{}".format(i, j), L.NStepBiLSTM(
                    1, inputdim, cdim, 0.0))
                if j == layers - 1:
                    _hdim = hdim * 2
                else:
                    _hdim = hdim
                setattr(self, "bt{}_l{}".format(i, j), L.Linear(2 * cdim, _hdim))

            _type = ['r', 'i']
            for i, j in combs2:
                name = 'filt{}_{}'.format(_type[j], i)
                _filt = L.Linear(2 * cdim, nchannels[i] * (nfft // 2 + 1))
                setattr(self, name, _filt)

            for i in range(len(nchannels)):
                name = 'brn_{}'.format(i)
                brn = L.BatchRenormalization(1)
                setattr(self, name, brn)

        self.channels = nchannels
        self.layers = layers
        fbanks = get_filterbanks(nfilt=nfilt).T
        self.fbanks = fbanks[None, :, :]
        self.nfilt = nfilt
        self.nfft = nfft

    def __call__(self, xs, ilens):
        # xs: [frame x channel(8) x dim(257)] x bs
        bs = len(xs)
        channels = xs[0].shape[1]
        ilens = np.array([int(ilens[i]) for i in range(bs)])
        idx = self.channels.index(channels // 2)
        xs1 = [xs[i].reshape(ilens[i], -1) for i in range(bs)]

        for layer in six.moves.range(self.layers):
            hy, cy, ys = self['bilstm{}_l{}'.format(idx, layer)](None, None, xs1)
            ys = self['bt{}_l{}'.format(idx, layer)](F.vstack(ys))
            xs1 = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
            del hy, cy
        # final tanh operation
        xs1 = F.vstack(xs1)
        rfilt = F.tanh(self['filtr_{}'.format(idx)](xs1))  # F.split_axis(, np.cumsum(ilens[:-1]), axis=0)
        # Conj of Filter
        ifilt = -1. * F.tanh(self['filti_{}'.format(idx)](xs1))
        min_range = chainer.Variable(self.xp.asarray([1e-20], dtype=np.float32))

        rxs, ixs = F.split_axis(F.vstack(xs), 2, axis=1)
        length, channels, dims = rxs.shape
        rxs = rxs.reshape(-1, channels * dims)
        ixs = ixs.reshape(-1, channels * dims)
        real_val = rfilt * rxs - ifilt * ixs
        imag_val = rfilt * ixs + ifilt * rxs
        xs = F.sum(real_val.reshape(-1, channels, dims), axis=1, keepdims=True) ** 2 + F.sum(
            imag_val.reshape(-1, channels, dims), axis=1, keepdims=True) ** 2
        xs = 1. / self.nfft * xs
        min_range = F.broadcast_to(min_range, xs.shape)
        fbanks = chainer.Variable(self.xp.asarray(self.fbanks, dtype=np.float32))
        fbanks = F.broadcast_to(fbanks, [length, self.nfft // 2 + 1, self.nfilt])
        xs = F.log10(F.matmul(F.maximum(xs, min_range), fbanks))
        xs = F.split_axis(xs, np.cumsum(ilens[:-1]), axis=0)
        xs = F.swapaxes(F.pad_sequence(xs), 1, 2)

        # The normalization is executed by a batch renorm  (Need to test with CMVN)
        xs = self['brn_{}'.format(idx)](xs)
        return xs, ilens


class RESLOC(chainer.Chain):
    def __init__(self, in_channel=1, mode=None, act=F.relu, bn=None,
                 outs=128, dropout=None, dratio=0.0, nopad=False, subsample='2222'):
        super(RESLOC, self).__init__()
        if type(in_channel) is int:
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(4)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(4)]
        elif mode == 'entry':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(2)] + [(0, y) for y in range(2, 4)]
        else:
            raise ValueError('Incorrect mode.')
        with self.init_scope():
            # CNN layer (RESNET motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 16, 1, stride=1, nobias=True)
                elif j == 1:
                    layer = BottleneckA(in_channel[i], 64, 64, act=act, bn=bn, groups=in_channel[i])
                elif j == 2:
                    layer = BottleneckA(16, 64, 64, act=act, bn=bn)
                else:
                    layer = BottleneckA(64, 128, outs, act=act, bn=bn)
                setattr(self, l_name, layer)

        for x in range(len(in_channel)):
            doutname = 'drop_{}'.format(x)
            douttype = no_dropout
            if in_channel[x] == 2:
                if dropout == 'inc':
                    logging.info('Adding Incremental dropout to the training')
                    douttype = dropout_incremental
                elif dropout == 'fix':
                    logging.info('Adding fixed dropout to the training')
                    douttype = dropout_fixed
                elif dropout == 'rand':
                    logging.info('Adding random dropout to the training')
                    douttype = dropout_random
            setattr(self, doutname, douttype)

        self.dropout = dropout
        self.in_channel = in_channel
        self.mode = mode
        self.iter = 0
        self.dratio = dratio
        self.nopad = nopad
        if subsample == '2222':
            self.subsample = self.subsample2222
        elif subsample == '3111':
            self.subsample = self.subsample3111
        else:
            raise ValueError('Incorrect type of subsample')

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        self.iter += 1
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        if not self.nopad:
            # x: utt x frame x input channel x dim
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)

        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        # Apply dropout only to the binaural input
        xs = self['drop_{}'.format(idx)](xs, self.dratio, self.iter)

        _xs = self['conv{}_1'.format(idx)](xs)
        xs = self['conv{}_0'.format(idx)](xs)
        
        if self.mode == 'entry':
            idx = 0

        xs = self['conv{}_2'.format(idx)](xs)
        xs = F.vstack(F.split_axis(xs, 16, axis=1))
        xs = F.split_axis(F.softmax(xs, axis=1), 16, axis=0)
        xs = _xs * F.concat(xs, axis=1)
        xs, ilens = self.subsample(idx, xs, ilens)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens

    def subsample2222(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = self['conv{}_3'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        return xs, ilens

    def subsample3111(self, idx, xs, ilens):
        xs = F.max_pooling_2d(xs, (3, 1), stride=(3, 1))

        xs = self['conv{}_3'.format(idx)](xs)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 3), dtype=np.int32)
        return xs, ilens


class DenseBlock(chainer.Chain):
    def __init__(self, in_ch, growth_rate, n_layer):
        self.n_layer = n_layer
        super(DenseBlock, self).__init__()
        for i in moves.range(self.n_layer):
            self.add_link('bn%d' % (i + 1),
                          L.BatchReNormalization(in_ch + i * growth_rate))
            self.add_link('conv%d' % (i + 1),
                          L.Convolution2D(in_ch + i * growth_rate, growth_rate,
                                          3, 1, 1))

    def __call__(self, x, dropout_ratio, train):
        for i in moves.range(1, self.n_layer + 1):
            h = F.relu(self['bn%d' % i](x, test=not train))
            h = F.dropout(self['conv%d' % i](h), dropout_ratio, train)
            x = F.concat((x, h))
        return x


class DENSENET(chainer.Chain):
    def __init__(self, in_channel=1, mode=None, act=F.relu, bn=None,
                 outs=128, dropout=None, dratio=0.0, nopad=False, subsample='2222', reshape=0):
        super(DENSENET, self).__init__()
        if type(in_channel) is int:
            in_channel = [in_channel]
        if mode == 'single':
            combs = [(0, y) for y in range(3)]
        elif mode == 'parallel':
            combs = [(x, y) for x in range(len(in_channel)) for y in range(3)]
        elif mode == 'entry':
            combs = [(x, 0) for x in range(len(in_channel))] + [(0, y) for y in range(1, 3)]
        else:
            raise ValueError('Incorrect mode.')
        with self.init_scope():
            # CNN layer (RESNET motivated)
            for i, j in combs:
                l_name = 'conv{}_{}'.format(i, j)
                if j == 0:
                    layer = L.Convolution2D(in_channel[i], 16, 1, stride=1)
                elif j == 1:
                    layer = DenseBlock(16, growth_rate=16, n_layer=4)
                else:
                    layer = DenseBlock(64, growth_rate=16, n_layer=4)
                setattr(self, l_name, layer)

        for x in range(len(in_channel)):
            doutname = 'drop_{}'.format(x)
            douttype = None
            if in_channel[x] == 2:
                if dropout == 'inc':
                    logging.info('Adding Incremental dropout to the training')
                    douttype = dropout_incremental
                elif dropout == 'fix':
                    logging.info('Adding fixed dropout to the training')
                    douttype = dropout_fixed
                elif dropout == 'rand':
                    logging.info('Adding random dropout to the training')
                    douttype = dropout_random
            setattr(self, doutname, douttype)

        self.dropout = dropout
        self.in_channel = in_channel
        self.mode = mode
        self.iter = 0
        self.dratio = dratio
        self.nopad = nopad
        self.reshape = reshape

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        self.iter += 1
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        if not self.nopad:
            # x: utt x frame x input channel x dim
            xs = F.pad_sequence(xs)

            # x: utt x input channel x frame x dim
            xs = F.swapaxes(xs, 1, 2)

        chn = xs.shape[1]
        idx = self.in_channel.index(chn)

        # Apply dropout only to the binaural input
        if not self['drop_{}'.format(idx)] is None:
            xs = self['drop_{}'.format(idx)](xs, self.dratio, self.iter)

        xs = self['conv{}_0'.format(idx)](xs)
        if self.mode == 'entry':
            idx = 0

        xs = self['conv{}_1'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = self['conv{}_2'.format(idx)](xs)
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)+
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens


class LMFILTPREV(chainer.Chain):
    def __init__(self, in_channel=1):
        super(LMFILTPREV, self).__init__()
        with self.init_scope():
            self.up = L.Convolution2D(in_channel, 16, (2, 1), stride=1, nobias=True)
            self.down = L.Convolution2D(16, in_channel, 1, stride=1, nobias=True)

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x input channel x dim
        xs = F.pad_sequence(xs)

        # x: utt x input channel x frame x dim
        xs = F.swapaxes(xs, 1, 2)

        _xs = F.relu(self.up(xs))
        xs = F.relu(xs[:, :, 1:] + self.down(_xs))

        # change ilens accordingly
        ilens = ilens - 1

        return xs, ilens


class LMFILTLT(chainer.Chain):
    def __init__(self, in_channel=1):
        super(LMFILTLT, self).__init__()
        with self.init_scope():
            self.left_up = L.Convolution2D(in_channel, 16, (2, 1), stride=1, nobias=True)
            self.left_down = L.Convolution2D(16, in_channel, 1, stride=1, nobias=True)
            self.right_up = L.Convolution2D(in_channel, 16, (2, 1), stride=1, nobias=True)
            self.right_down = L.Convolution2D(16, in_channel, 1, stride=1, nobias=True)

    def __call__(self, xs, ilens):
        '''RESNET forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x input channel x dim
        xs = F.pad_sequence(xs)

        # x: utt x input channel x frame x dim
        xs = F.swapaxes(xs, 1, 2)

        l_xs = self.left_down(F.relu(self.left_up(xs[:, :, :-1])))
        r_xs = self.right_down(F.relu(self.right_up(xs[:, :, 1:])))
        with chainer.no_backprop_mode():
            k = r_xs / l_xs
        xs = F.relu(k * l_xs + xs[:, :, 1:-1] + (1 - k) * r_xs)

        # change ilens accordingly
        ilens = ilens - 2

        return xs, ilens

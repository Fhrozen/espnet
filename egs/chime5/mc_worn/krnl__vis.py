#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import platform
import random
import subprocess
import sys

import numpy as np
import logging
import math
import sys

import numpy as np
import six

import torch
from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import Loss


def main():
    parser = argparse.ArgumentParser()
    # general configuration

    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
 
    # network archtecture
    # encoder
    parser.add_argument('--minput', default='parallel', type=str,
                        help='Mode of input processing')
    parser.add_argument('--eintype', default=None, type=str,
                        help='Input of encoder configuration')
    parser.add_argument('--eunits', '-u', default=512, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=512, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default=1, type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')
    # loss
    parser.add_argument('--ctc_type', default='warpctc', type=str,
                        choices=['chainer', 'warpctc'],
                        help='Type of CTC implementation to calculate loss.')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                 'coverage_location', 'location2d', 'location_recurrent',
                                 'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                 'multi_head_multi_res_loc'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=320, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--awin', default=5, type=int,
                        help='Window size for location2d attention')
    parser.add_argument('--aheads', default=4, type=int,
                        help='Number of heads for multi head attention')
    parser.add_argument('--aconv-chans', default=-1, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--aconv-filts', default=100, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    # decoder
    parser.add_argument('--dtype', default='lstm', type=str,
                        choices=['lstm'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=300, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--mtlalpha', default=0.5, type=float,
                        help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
    parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                        help='Apply label smoothing with a specified distribution type')
    parser.add_argument('--lsm-weight', default=0.0, type=float,
                        help='Label smoothing weight')

    # optimization related
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    args.elayers = 2
    args.etype = 'resblstmp'
    args.einputs = [4, 2]
    args.char_list = None
    args.dropout_rate = 0.0
    args.atype = 'location'
    args.aconv_chans = 10
    args.outdir = './exp/train_worn_u25k_backend_pytorch_resblstmp_e2_subsample1_2_2_1_1_modeparallel_unit512_proj512_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.1_adadelta_bs30_mli800_mlo150_lsmunigram0.05'

    args.model = os.path.join(args.outdir, 'results', 'model.acc.best')
    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])
    e2e = E2E(83, 44, args)
    model = Loss(e2e, args.mtlalpha)
    def cpu_loader(storage, location):
        return storage

    def remove_dataparallel(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        return new_state_dict

    model.load_state_dict(remove_dataparallel(torch.load(args.model, map_location=cpu_loader)))
    
    weights = np.asarray(model.predictor.enc.enc1.resblock1_2.conv1.weight.data)
    dims = weights.shape
    _weights = weights.reshape(dims[0] * dims[1], dims[2] * dims[3])
    print(_weights.shape)

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import decomposition

    comps = 3
    embedder = decomposition.PCA(n_components=comps, svd_solver='randomized')
    reduced = embedder.fit_transform(_weights)

    plt.figure()
    plt.imshow(_weights)

    # plt.figure()
    # plt.scatter(reduced[:,0], reduced[:,1])
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2])

    plt.show()
    return
    
    


if __name__ == '__main__':
    main()

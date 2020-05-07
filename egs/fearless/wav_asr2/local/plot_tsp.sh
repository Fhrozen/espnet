#!/bin/bash


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


expdir=exp/train_chainer_train_transf_wave_res_learn_feats_wave_frames

./local/plot_tsp.py \
            --snapshots ${expdir}/results/models/model.ep.* \
            --out ${expdir}/results/tsps

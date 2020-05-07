#!/bin/bash


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


expdir=exp/train_chainer_train_transf_sinc_feats_wave

./local/save_model.py \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/models/${recog_model}

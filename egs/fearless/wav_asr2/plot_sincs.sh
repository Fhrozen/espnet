#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


expdir=exp/train_chainer_train_transf_sinc_feats_wave

./local/plot_feats_sinc.py \
            --snapshots ${expdir}/results/models/model.ep.* \
            --out ${expdir}/results/feats

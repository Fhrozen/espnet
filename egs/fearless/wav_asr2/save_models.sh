#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


expdir=exp/train_chainer_train_transf_sinc_feats_wave

./local/save_model.py \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/models/${recog_model}

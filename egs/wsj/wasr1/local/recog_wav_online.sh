#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration

# feature configuration
do_delta=false

# config files
preprocess_config=conf/preprocess_stft.json  # use conf/specaug.yaml for data augmentation
train_config=conf/tuning/train_2.yaml
decode_config=conf/tuning/decode_online.yaml

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

expdir=exp/train_si284_chainer_train_2_preprocess_stft
# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo "Decoding"

if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
    recog_model=model.last${n_average}.avg.best${n_average}
fi

ngpu=0
backend=chainer
local/online_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --result-label ${expdir}/online_decode \
        --backend ${backend} \
        --model ${expdir}/results/${recog_model}
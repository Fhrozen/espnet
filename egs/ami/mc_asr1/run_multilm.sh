#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=-1       # start from -1 if you need to start from data download
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# network archtecture
# encoder related
einputs=8
etype=vgg_blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=0 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=20000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=64    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs
use_lm=true


lmch_dir="exp/rnnlm/train_chainer_2layer_unit1024_adam0.001_bs128_maxlen200"
lmwd_dir="exp/rnnlm/train_chainer_1layer_unit1024_sgd1_bs512_maxlen100_word4000"


# decoding parameter
lm_weight=0.5
beam_size=20
penalty=0.2
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
njobs=32

# scheduled sampling option
samp_prob=0.0


# exp tag
# tag="" # tag for managing experiments.
expdir=
. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Path where AMI gets downloaded (or where locally available):

recog_set="mdm8_dev mdm8_eval ihm_eval ihm_dev"


# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)

if [ -z ${lmch_dir} ]; then
    echo "Need to specify a Character-LM"
    exit 1
fi
if [ -z ${lmwd_dir} ]; then
    echo "Need to specify a Word-LM"
    exit 1
fi

if [ -z ${expdir} ]; then
    echo "Need to specify the trained model"
    exit 1
fi


if ! [ -d ${lmch_dir} ]; then
    if [ -d ../asr1/${lmch_dir} ]; then
        cp -r ../asr1/${lmch_dir} ./${lmch_dir}
    else
        echo "First need to train ${lmch_dir} in ../asr1"
        exit 1
    fi
fi
if ! [ -d ${lmwd_dir} ]; then
    if [ -d ../asr1/${lmwd_dir} ]; then
        cp -r ../asr1/${lmwd_dir} ./${lmwd_dir}
    else
        echo "First need to train ${lmwd_dir} in ../asr1"
        exit 1
    fi
fi

dict=data/lang_1char/mdm8_ihm_train_units.txt
echo "dictionary: ${dict}"

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=${njobs}

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        decode_dir=${decode_dir}_multilvl_lm${lm_weight}
        recog_opts="--word-rnnlm ${lmwd_dir}/rnnlm.model.best"
        recog_opts="${recog_opts} --rnnlm ${lmch_dir}/rnnlm.model.best"

        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            $recog_opts &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


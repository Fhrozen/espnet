#!/bin/bash

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network archtecture
# encoder related
einputs=4
etype=vgg_blstmp     # encoder architecture type
elayers=3
eunits=512
eprojs=512
subsample=0 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=512
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.1

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=25
maxlen_in=750  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
use_wordlm=false     # false means to train/use a character LM
lm_vocabsize=35000  # effective only for word LMs
lm_lr=1             # training lr for LMs (sgd)
lm_alpha=0.001     # training alpha for LMs (adam)
lm_layers=2         # 2 for character LMs
lm_units=650        # 650 for character LMs
lm_opt=adam          # adam for character LMs
lm_batchsize=1024   # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=100       # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

lmwd_dir="exp/rnnlm/train_chainer_1layer_unit512_sgd1_bs512_maxlen100_word4000"
lmch_dir="exp/rnnlm/train_chainer_2layer_unit1024_adam0.001_bs128_maxlen200"

# decoding parameter
lm_weight=0.1
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best
njobs=32


# scheduled sampling option
samp_prob=0.0

# data
train_set=train_mix_worn_uall  # train_uall train_mix_white_worn_uall

# exp tag
expdir=

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
recog_set="dev_worn dev_ref"  # dev_wpe_ref eval_ref eval_wpe_ref  


dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"

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
    fi
fi
if ! [ -d ${lmwd_dir} ]; then
    if [ -d ../asr1/${lmwd_dir} ]; then
        cp -r ../asr1/${lmwd_dir} ./${lmwd_dir}
    else
        echo "First need to train ${lmwd_dir} in ../asr1"
    fi
fi


if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=${njobs}

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
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
            --debugmode ${debugmode} \
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

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


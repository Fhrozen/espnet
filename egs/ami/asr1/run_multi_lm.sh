#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
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
etype=blstmp     # encoder architecture type
elayers=8
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

# decoding parameter
lm_weight=0.5
beam_size=20
penalty=0.2
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm

# exp tag
tag="" # tag for managing experiments.
train_subset=train
. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,

train_set=${mic}_${train_subset}
train_dev=${mic}_dev
train_test=${mic}_eval
recog_set="${mic}_dev ${mic}_eval"

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

if [[ ${stage} -le 3 && $use_lm == true ]]; then
    echo "stage 3: LM Preparation"
    if [ $use_wordlm = true ]; then
	lmdatadir=data/local/wordlm_train
	lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
	mkdir -p ${lmdatadir}
        cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train.txt
        cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
	lmdatadir=data/local/lm_train
	lmdict=$dict
	mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " \
            > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
            > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 data/${train_test}/text | cut -f 2- -d" " \
            > ${lmdatadir}/test.txt
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi


if [[ ${stage} -le 3 && $use_lm == true ]]; then
    echo "stage 3: LM Preparation"
    if [ $use_wordlm = true ]; then
	lmdatadir=data/local/wordlm_train
	lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
	mkdir -p ${lmdatadir}
        cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train.txt
        cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
	lmdatadir=data/local/lm_train
	lmdict=$dict
	mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " \
            > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
            > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 data/${train_test}/text | cut -f 2- -d" " \
            > ${lmdatadir}/test.txt
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_ngpu${ngpu}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi



if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_lmmultilevel_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        if [ $use_lm = true ]; then
            decode_dir=${decode_dir}_rnnlm${lm_weight}_${lmtag}
            if [ $use_wordlm = true ]; then
                recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi
        else
            echo "No language model is involved."
            recog_opts=""
        fi

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


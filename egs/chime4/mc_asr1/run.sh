#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

# network architecture
# encoder related
einputs=6
etype=vgg_blstmp     # encoder architecture type
elayers=3
eunits=1024
eprojs=1024
subsample=0 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=1024
# attention related
atype=location
adim=1024
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=20
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=10

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=20
penalty=0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
chime4_data=${CHIME4_CORPUS}
wsj0=${WSJ0_CORPUS}
wsj1=${WSJ1_CORPUS}

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

train_set=tr05_mix_multi_noisy # tr05_mix_multi_noisy_si284
train_dev=dt05_simu_noisy_6mics
recog_set="\
dt05_real_noisy_2mics dt05_simu_noisy_2mics et05_real_noisy_2mics et05_simu_noisy_2mics \
dt05_real_noisy_6mics dt05_simu_noisy_6mics et05_real_noisy_6mics et05_simu_noisy_6mics \
"
if [ ${stage} -le -1 ]; then
    echo "stage -1: Prepare 2CH training data"
    local/prepare_2ch_tr.sh ${chime4_data}
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    
    echo "prepartion for chime4 data 2CH"
    local/real_noisy_chime4_data_prep.sh ${chime4_data} isolated_2ch_track 2mics
    local/simu_noisy_chime4_data_prep.sh ${chime4_data} isolated_2ch_track 2mics
    echo "prepartion for chime4 data 6CH"
    local/real_noisy_chime4_data_prep.sh ${chime4_data} isolated 6mics
    local/simu_noisy_chime4_data_prep.sh ${chime4_data} isolated 6mics
    
    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    # Copy WSJ0-1 from CHiME4/ASR1
    utils/copy_data_dir.sh ../asr1/data/tr05_multi_noisy_si284 data/tr05_multi_noisy_si284 
    # utils/combine_data.sh data/tr05_multi_noisy_si284 data/tr05_real_noisy_6mics data/train_si284 data/tr05_simu_noisy_6mics
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    fbankdir=fbank
    tr_set="train_si284 tr05_real_noisy_2mics tr05_simu_noisy_2mics tr05_real_noisy_6mics tr05_simu_noisy_6mics"
    tasks="${tr_set} ${recog_set}"
    for x in ${tasks}; do
        utils/copy_data_dir.sh data/${x} data-fbank/${x}
        utils/copy_data_dir.sh data/${x} data-stft/${x}
        steps/make_fbank_pitch.sh --nj 8 --cmd "${train_cmd}" data-fbank/${x} exp/make_fbank/${x} ${fbankdir}
    done

    echo "combine real and simulation data"
    utils/combine_data.sh data-fbank/tr05_multi_noisy_2mics data-fbank/tr05_simu_noisy_2mics data-fbank/tr05_real_noisy_2mics
    utils/combine_data.sh data-fbank/tr05_multi_noisy_6mics data-fbank/tr05_simu_noisy_6mics data-fbank/tr05_real_noisy_6mics

    # compute global CMVN
    mkdir -p data-fbank/${train_set}
    compute-cmvn-stats scp:data-fbank/tr05_multi_noisy_6mics/feats.scp data-fbank/${train_set}/cmvn.ark

    # dump features for training
    echo "dump features"
    for ttask in 2mics 6mics; do
        feat_tr=${dumpdir}/tr05_multi_noisy_${ttask}/delta${do_delta}; mkdir -p ${feat_tr}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data-fbank/tr05_multi_noisy_${ttask}/feats.scp data-fbank/${train_set}/cmvn.ark exp/dump_feats/train/${ttask} \
            ${feat_tr}
    done

    feat_tr=${dumpdir}/train_si284/delta${do_delta}; mkdir -p ${feat_tr}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data-fbank/train_si284/feats.scp data-fbank/train_si284/cmvn.ark exp/dump_feats/train/train_si284 \
        ${feat_tr}
    
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data-fbank/${rtask}/feats.scp data-fbank/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data-fbank/tr05_multi_noisy_2mics/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data-fbank/tr05_multi_noisy_2mics/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for ttask in 2mics 6mics; do
        feat_tr=${dumpdir}/tr05_multi_noisy_${ttask}/delta${do_delta}
        data2json.sh --multi 1 --feat ${feat_tr}/feats.scp --nlsyms ${nlsyms} \
            data-fbank/tr05_multi_noisy_${ttask} ${dict} > ${feat_tr}/data.json
    done
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --multi 1 --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}

if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    [[ -d ${lmexpdir} ] || ls -l ${lmexpdir}] && ln -s ../asr1/${lmexpdir} ./exp || exit 1
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_m${emode}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    if [ "${emode}" == "regular" ]; then
        tr_set="tr05_multi_noisy_6mics"
    else
        tr_set="tr05_multi_noisy_2mics tr05_multi_noisy_6mics"
    fi
    tfiles=""
    for tset in ${tr_set}; do
        tfiles="${dumpdir}/${tset}/delta${do_delta}/data.json ${tfiles}"
    done
    joinjson.py ${tfiles} > ${feat_tr_dir}/data.json    
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --einputs ${einputs} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --converter mcbank
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=8

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        if [ $use_wordlm = true ]; then
            decode_dir=${decode_dir}_wordrnnlm${lm_weight}
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best --word-dict ${lmdict}"
        else
            decode_dir=${decode_dir}_rnnlm${lm_weight}
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
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


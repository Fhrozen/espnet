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

# decoding parameter
lm_weight=0.1
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
chime5_corpus=${CHIME5_CORPUS}
train_set=train_mix_worn_uall  # train_uall train_mix_white_worn_uall

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

json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio
train_dev=dev_ref
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#recog_set="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"
recog_set="dev_worn dev_ref dev_wpe_ref" # eval_ref eval_wpe_ref"  
noises="white"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Augmentation"
    for noise in white; do 
        local/augmentation.py --folder ${audio_dir}/train_${noise} \
            --noise-type ${noise} \
            --audio-folder ${audio_dir}/train
    done
fi

if [ ${stage} -le 0 ]; then
    ### The Code employs the data prepared in single channel 
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data files preparation"
    for noise in ${noises}; do
        trainset=train
        if [ "${noise}" != "None" ]; then
            trainset="${trainset}_${noise}"
        fi
        for mictype in worn u01 u02 u04 u05 u06; do
            local/prepare_data.sh --mictype ${mictype} --noise ${noise} \
                    ${audio_dir}/${trainset} ${json_dir}/train data/${trainset}_${mictype}
        done
        utils/combine_data.sh data/${trainset}_uall data/${trainset}_u01 data/${trainset}_u02 data/${trainset}_u04 data/${trainset}_u05 data/${trainset}_u06
        rm -rf data/${trainset}_u0*
    done

    utils/copy_data_dir.sh ../asr1/data/${dset}_worn_stere  data/${dset}_worn 
fi

trainsets="train train_white"
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta};
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    folders=""
    for x in uall worn ; do
        for y in ${trainsets}; do
            folders="${y}_${x} ${folders}"
        done
    done
    for x in ${folders} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    folders=""
    for x in uall worn ; do
        for y in ${trainsets}; do
            folders="data/${y}_${x} ${folders}"
        done
    done
    utils/combine_data.sh data/${train_set} ${folders}
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
fi
dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # remove 1 or 0 length outputs
    #utils/copy_data_dir.sh data/train_worn_u200k data/train_worn_u200k_org
    #remove_longshortdata.sh --nlsyms ${nlsyms} --minchars 1 data/train_worn_u200k_org data/train_worn_u200k

    folders=""
    for x in uall worn ; do
        for y in ${trainsets}; do
            folders="${y}_${x} ${folders}"
        done
    done
    for rtask in ${folders}; do
        feat_tr=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_tr}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_tr}
    done

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_recog_dir}
    done

    echo "make json files"
    for rtask in ${folders}; do
        feat_tr=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_tr}
        data2json.sh --multi 1 --feat ${feat_tr}/feats.scp --nlsyms ${nlsyms} \
            data/${rtask} ${dict} > ${feat_tr}/data.json
    done
    
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --multi 1 --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
    
    feat_tr=${dumpdir}/train_mix_worn_uall/delta${do_delta}; mkdir -p ${feat_tr}
    joinjson.py ${dumpdir}/train_worn/delta${do_delta}/data.json \
                ${dumpdir}/train_uall/delta${do_delta}/data.json  > ${feat_tr}/data.json  

    feat_tr=${dumpdir}/train_mix_white_worn_uall/delta${do_delta}; mkdir -p ${feat_tr}
    joinjson.py ${dumpdir}/train_mix_worn_uall/delta${do_delta}/data.json \
                ${dumpdir}/train_white_worn/delta${do_delta}/data.json > ${feat_tr}/data.json  
    exit 0
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ "${lm_opt}" == "sgd" ]; then
    lm_optval=${lm_lr}
    lm_options="--lr ${lm_lr}"
else
    lm_optval=${lm_alpha}
    lm_options="--alpha ${lm_alpha}"
fi
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}${lm_optval}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}

if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    if [ $use_wordlm = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cat data/train_worn/text | cut -f 2- -d" " | sort  | sed 'n; d' > ${lmdatadir}/train.txt
        cat data/${train_dev}/text | cut -f 2- -d" "  > ${lmdatadir}/valid.txt
        cat ${lmdatadir}/train.txt ${lmdatadir}/valid.txt > ${lmdatadir}/train_valid.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train_valid.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=$dict
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/train_worn/text \
            | cut -f 2- -d" " > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
    fi
    # This will copy the LM from the Single Channel
    # The languange model is trained only once
    if ! [ -d ${lmexpdir} ]; then
        if [ -d ../asr1/${lmexpdir} ]; then
            cp -r ../asr1/${lmexpdir} ./${lmexpdir}
        else
            mkdir -p ${lmexpdir}
            # use only 1 gpu
            if [ ${ngpu} -gt 1 ]; then
                echo "LM training does not support multi-gpu. single gpu will be used."
            fi
            ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
            lm_train.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --verbose 1 \
            --outdir ${lmexpdir} \
            --train-label ${lmdatadir}/train.txt \
            --valid-label ${lmdatadir}/valid.txt \
            --resume ${lm_resume} \
            --layer ${lm_layers} \
            --unit ${lm_units} \
            --opt ${lm_opt} \
            --batchsize ${lm_batchsize} \
            --epoch ${lm_epochs} \
            --maxlen ${lm_maxlen} \
            --dict ${lmdict} \
            ${lm_options}
        fi
    fi
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --converter mcbank \
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
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs}
    exit 0
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=64

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        if [ $use_wordlm = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
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


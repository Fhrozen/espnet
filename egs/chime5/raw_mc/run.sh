#!/bin/bash

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

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
lm_weight=0.1
use_wordlm=false
vocabsize=65000

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=model.acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
chime5_corpus=${CHIME5_CORPUS}
datanoise=None

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
train_set=train_mix_worn_uall
train_dev=dev_ref
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#recog_set="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"
recog_set="dev_worn dev_ref dev_wpe_ref" # eval_ref eval_wpe_ref"  
noises="None white"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Augmentation"
    for noise in white; do 
        local/augmentation.py --folder ${audio_dir}/train_${noise} \
            --noise-type ${noise} \
            --audio-folder ${audio_dir}/train
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data files preparation"
    for noise in ${noises}; do
        trainset=train
        label=train
        if [ "${noise}" != "None" ]; then
            label="${trainset}_${noise}"
            trainset="${noise}/${trainset}"
        fi
        for mictype in worn u01 u02 u04 u05 u06; do
            local/prepare_data.sh --mictype ${mictype} --noise ${noise} \
                    ${audio_dir}/${trainset} ${json_dir}/train data/${label}_${mictype}
        done
        utils/combine_data.sh data/${label}_uall data/${label}_u01 data/${label}_u02 data/${label}_u04 data/${label}_u05 data/${label}_u06
        rm -rf data/${label}_u0*
    done

    utils/combine_data.sh data/train_worn_uall data/train_worn data/train_uall

    for dset in dev; do
    for mictype in worn; do
        local/prepare_data.sh --mictype ${mictype} \
                ${audio_dir}/${dset} ${json_dir}/${dset} \
                data/${dset}_${mictype}
    done
    done

    for dset in dev eval; do
        for aset in "" "/wpe"; do
            local/prepare_data.sh --mictype ref \
                ${audio_dir}${aset}/${dset} ${json_dir}/${dset} \
                data/${dset}${aset////_}_ref
        done
    done
fi
trainsets=""
for noise in ${noises}; do
    trainset=train
    if [ "${noise}" != "None" ]; then
        trainset="${trainset}_${noise}"
    fi
    trainsets="${trainset} ${trainsets}"
done

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    folders=""
    for x in uall worn ; do
        for y in ${trainsets}; do
            folders="${y}_${x} ${folders}"
        done
    done
    for x in ${folders}; do
        local/make_feats.sh --cmd "$train_cmd" --compress false --type 1 --nj 20 data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    folders=""
    for x in uall worn ; do
        for y in ${trainsets}; do
            folders="data/${y}_${x} ${folders}"
        done
    done
    # compute global CMVN
    utils/combine_data.sh data/${train_set} ${folders}
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
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ $use_wordlm = true ]; then
    lmdatadir=data/local/wordlm_train
    lm_batchsize=256
    lmexpdir=exp/train_rnnlm_${backend}_word_2layer_bs${lm_batchsize}
    lmdict=${lmexpdir}/wordlist_${vocabsize}.txt
else
    lmdatadir=data/local/lm_train
    lm_batchsize=2048
    lmexpdir=exp/train_rnnlm_${backend}_2layer_bs${lm_batchsize}
    lmdict=$dict
fi

mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    mkdir -p ${lmdatadir}
    if [ $use_wordlm = true ]; then
	cat data/train_worn_uall/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
							    > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
	cat data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
							    > ${lmdatadir}/valid.txt
	text2vocabulary.py -s ${vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
    text2token.py -s 1 -n 1 -l ${nlsyms} data/train_worn_uall/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. single gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
		--batchsize ${lm_batchsize} \
		--dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_n${datanoise}_subsample${subsample}_mode${emode}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
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
    if [ "${emode}" == "regular" ]; then
        cp ${dumpdir}/train_uall/delta${do_delta}/data.json ${feat_tr_dir}/data.json
    else
        #TODO: needs to add subset if necessary
        if [ "${datanoise}" == "None" ]; then
            trainsets=train
        fi
        dfiles=""
        for dset in ${trainsets}; do
            dfiles="${dumpdir}/${dset}_worn/delta${do_delta}/data.json ${dfiles}"
        done
        for dset in train; do
            dfiles="${dumpdir}/${dset}_uall/delta${do_delta}/data.json ${dfiles}"
        done
        joinjson.py ${dfiles} > ${feat_tr_dir}/data.json    
    fi
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
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --converter mcspec
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=2

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


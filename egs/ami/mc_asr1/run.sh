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


# exp tag
tag="" # tag for managing experiments.
train_set=mdm8_train  # mdm8_ihm_train for 8 + 1 CH
. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
    clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
esac

train_dev=mdm8_dev
train_test=mdm8_eval
recog_set="mdm8_dev mdm8_eval ihm_eval"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
	echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
	echo " ... Assuming the data does not need to be downloaded.  Please use --stage 0 or more."
	exit 1
    fi
    mic=mdm8
    if [ -e data/local/downloads/wget_$mic.sh ]; then
	echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
    exit 1
    fi
    mic=ihm
    if [ -e data/local/downloads/wget_$mic.sh ]; then
	echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
	exit 1
    fi
    local/ami_download.sh $mic $AMI_DIR
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # common data prep
    if [ ! -d data/local/downloads/annotations ]; then
	local/ami_text_prep.sh data/local/downloads
    fi
    mic=mdm8
    base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
	PROCESSED_AMI_DIR=$AMI_DIR
    local/ami_${base_mic}_mc_data_prep.sh $PROCESSED_AMI_DIR $mic
    local/ami_${base_mic}_mc_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic dev
    local/ami_${base_mic}_mc_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic eval
    for dset in train dev eval; do
	# changed the original AMI data structure in the Kaldi recipe to the following
	utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 data/${mic}/${dset}_orig data/${mic}_${dset}
    done
    mic=ihm
    base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
    local/ami_${base_mic}_data_prep.sh $PROCESSED_AMI_DIR $mic
    local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic eval
    for dset in train eval; do
	# changed the original AMI data structure in the Kaldi recipe to the following
	utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 data/${mic}/${dset}_orig data/${mic}_${dset}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    mic=mdm8
    for x in ${mic}_train ${mic}_dev ${mic}_eval; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
    mic=ihm
    for x in ${mic}_train ${mic}_eval; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    utils/combine_data.sh data/mdm8_ihm_train data/mdm8_train data/ihm_train
    # compute global CMVN
    compute-cmvn-stats scp:data/mdm8_ihm_train/feats.scp data/mdm8_ihm_train/cmvn.ark

    # dump features for training
    for rtask in mdm8 ihm; do
        feat_tr=${dumpdir}/${rtask}_train/delta${do_delta}; mkdir -p ${feat_tr}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}_train/feats.scp data/mdm8_ihm_train/cmvn.ark exp/dump_feats/train ${feat_tr}
    done
    
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/mdm8_ihm_train/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
            data/${rtask}/feats.scp data/mdm8_ihm_train/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    for rtask in mdm8 ihm; do
        multi=1
        if [ "${rtask}" == "ihm" ]; then
            multi=0
        fi
        feat_tr=${dumpdir}/${rtask}_train/delta${do_delta}
        data2json.sh --multi ${multi} --feat ${feat_tr}//feats.scp \
            data/${rtask}_train ${dict} > ${feat_tr}/data.json
    done

    mkdir -p ${dumpdir}/mdm8_ihm_train/delta${do_delta}
    joinjson.py ${dumpdir}/ihm_train/delta${do_delta}/data.json \
                ${dumpdir}/mdm8_train/delta${do_delta}/data.json > ${dumpdir}/mdm8_ihm_train/delta${do_delta}/data.json 
    data2json.sh --multi 1 --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        multi=1
        if [ "${rtask}" == "ihm_eval" ]; then
            multi=0
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --multi ${multi} --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
    exit 0
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
mkdir -p ${lmexpdir}

if [[ ${stage} -le 3 && $use_lm == true ]]; then
    echo "stage 3: LM Preparation"
    if [ $use_wordlm = true ]; then
	lmdatadir=data/local/wordlm_train
	lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
	mkdir -p ${lmdatadir}
        cat data/ihm_train/text | cut -f 2- -d" " > ${lmdatadir}/train.txt
        cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
	lmdatadir=data/local/lm_train
	lmdict=$dict
	mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 data/ihm_train/text | cut -f 2- -d" " \
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
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
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
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
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
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

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


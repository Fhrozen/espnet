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
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.1

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
lm_weight=0.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
datasize=100 # in K
chime5_corpus=${CHIME5_CORPUS}
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_u${datasize}k
devsize=10 # in k
train_dev=dev_u${devsize}k_ref
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#recog_set="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"
recog_set="dev_ref dev_wpe_ref"  

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for mictype in worn u01 u02 u04 u05 u06; do
        local/prepare_data.sh --mictype ${mictype} \
			      ${audio_dir}/train ${json_dir}/train data/train_${mictype}
    done

    # combine mix array and worn mics
    # randomly extract first #k utterances from all mics
    # if you want to include more training data, you can increase the number of array mic utterances
    utils/combine_data.sh data/train_uall data/train_u01 data/train_u02 data/train_u04 data/train_u05 data/train_u06
    # use ch 1 to split
    utils/copy_data_dir.sh data/train_uall data/train_uall_multich/CH1
    grep "\.CH1-" data/train_uall/text > data/train_uall_multich/CH1/text
    utils/fix_data_dir.sh data/train_uall_multich/CH1
    utils/subset_data_dir.sh data/train_uall_multich/CH1 $((${datasize} * 1000)) data/train_u${datasize}k_per_ch/CH1
    rm -rf data/train_uall_multich
    for CH in 2 3 4; do
        utils/copy_data_dir.sh data/train_uall data/train_u${datasize}k_per_ch/CH${CH}
        sed -e "s/.CH1-/.CH${CH}-/g" data/train_u${datasize}k_per_ch/CH1/text > data/train_u${datasize}k_per_ch/CH${CH}/text
        utils/fix_data_dir.sh data/train_u${datasize}k_per_ch/CH${CH}
    done
    utils/combine_data.sh data/train_u${datasize}k data/train_u${datasize}k_per_ch/CH1 data/train_u${datasize}k_per_ch/CH2 data/train_u${datasize}k_per_ch/CH3 data/train_u${datasize}k_per_ch/CH4
    
    worn_size=25 # in K

    # use ch L to split
    utils/copy_data_dir.sh data/train_worn data/train_worn_multich/L
    grep "\.L-" data/train_worn/text > data/train_worn_multich/L/text
    utils/fix_data_dir.sh data/train_worn_multich/L
    utils/subset_data_dir.sh data/train_worn_multich/L $((${worn_size} * 1000)) data/train_worn_u${worn_size}k_per_ch/L
    rm -rf data/train_worn_multich
    for CH in R; do
        utils/copy_data_dir.sh data/train_worn data/train_worn_u${worn_size}k_per_ch/${CH}
        sed -e "s/.L-/.${CH}-/g" data/train_worn_u${worn_size}k_per_ch/L/text > data/train_worn_u${worn_size}k_per_ch/${CH}/text
        utils/fix_data_dir.sh data/train_worn_u${worn_size}k_per_ch/${CH}
    done
    utils/combine_data.sh data/train_worn_u${worn_size}k data/train_worn_u${worn_size}k_per_ch/L data/train_worn_u${worn_size}k_per_ch/R   
    utils/combine_data.sh data/train_worn_uall data/train_worn data/train_uall

    for dset in dev; do
	for mictype in u01 u02 u04 u03 u06; do
	    local/prepare_data.sh --mictype ${mictype} \
				  ${audio_dir}/${dset} ${json_dir}/${dset} \
				  data/${dset}_${mictype}
	done
    done
    utils/combine_data.sh data/dev_ref data/dev_u01 data/dev_u02 data/dev_u04 data/dev_u03 data/dev_u06
    utils/copy_data_dir.sh data/dev_ref data/dev_ref_multich/CH1
    grep "\.CH1-" data/dev_ref/text > data/dev_ref_multich/CH1/text
    utils/fix_data_dir.sh data/dev_ref_multich/CH1
    utils/subset_data_dir.sh data/dev_ref_multich/CH1 $((${devsize} * 1000)) data/dev_u${devsize}k_ref_per_ch/CH1
    rm -rf data/dev_ref_multich
    for CH in 2 3 4; do
        utils/copy_data_dir.sh data/dev_ref data/dev_u${devsize}k_ref_per_ch/CH${CH}
        sed -e "s/.CH1-/.CH${CH}-/g" data/dev_u${devsize}k_ref_per_ch/CH1/text > data/dev_u${devsize}k_ref_per_ch/CH${CH}/text
        utils/fix_data_dir.sh data/dev_u${devsize}k_ref_per_ch/CH${CH}
    done
    utils/combine_data.sh data/dev_u${devsize}k_ref data/dev_u${devsize}k_ref_per_ch/CH1 data/dev_u${devsize}k_ref_per_ch/CH2 data/dev_u${devsize}k_ref_per_ch/CH3 data/dev_u${devsize}k_ref_per_ch/CH4

    for dset in dev_wpe; do
	for mictype in u01 u02 u04 u03 u06; do
	    local/prepare_data.sh --mictype ${mictype} \
				  ${audio_dir}/${dset} ${json_dir}/dev \
				  data/${dset}_${mictype}
	done
    done
    utils/combine_data.sh data/dev_wpe_ref data/dev_wpe_u01 data/dev_wpe_u02 data/dev_wpe_u04 data/dev_wpe_u03 data/dev_wpe_u06
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
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

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/chime5/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/chime5/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_recog_dir}
    done

    echo "make json files"
    data2json.sh --multi 1 --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --multi 1 --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --batchsize 256 \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
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
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --einputs ${einputs} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=1

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        data=data/${rtask}
        #split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # make json labels for recognition
        #for j in `seq 1 ${nj}`; do
        #    data2json.sh --multi 1 --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} \
        #        ${sdata}/${j} ${dict} > ${sdata}/${j}/data.json
        #done

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${sdata}/JOB/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


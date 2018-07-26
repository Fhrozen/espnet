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
emode=regular
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
lm_weight=0.1

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
datasize=350 # in K
worn_size=75 # in K
chime5_corpus=${CHIME5_CORPUS}

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

json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio
train_set=train_mix_worn_u${datasize}k
train_dev=dev_ref
# use the below once you obtain the evaluation data. Also remove the comment #eval# in the lines below
#eval#recog_set="dev_worn dev_${enhancement}_ref eval_${enhancement}_ref"
recog_set="dev_worn dev_ref dev_wpe_ref eval_ref eval_wpe_ref"  
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

    utils/combine_data.sh data/train_worn_uall data/train_worn data/train_uall

    for dset in dev; do
    for mictype in worn; do
        local/prepare_data.sh --mictype ${mictype} \
                ${audio_dir}/${dset} ${json_dir}/${dset} \
                data/${dset}_${mictype}
    done
    done

    for dset in dev dev_wpe; do
        local/prepare_data.sh --mictype ref \
        ${audio_dir}/${dset} ${json_dir}/dev \
        data/${dset}_ref
    done

    for eset in "eval eval_wpe"; do
        local/prepare_data.sh --mictype ref \
        ${audio_dir}/${eset} ${json_dir}/eval \
        data/${eset}_ref
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


if [ ${stage} -le 1 ]; then
    echo "stage 1: (0.1) Data files split"
    # It is probably that this stage will be deleted and 
    # try to prepare all the dataset and split only the requested amount
    # for the training

    # use ch 1 to split
    for trainset in ${trainsets}; do
        utils/copy_data_dir.sh data/${trainset}_uall data/${trainset}_u${datasize}k_per_ch/CH1
        grep "\.CH1-" data/${trainset}_uall/text > data/${trainset}_u${datasize}k_per_ch/CH1/text
        echo "Subset file"
        local/subset.py --text data/${trainset}_u${datasize}k_per_ch/CH1/text --outfolder data/${trainset}_u${datasize}k_per_ch/CH1 --utters $((${datasize} * 1000))
        utils/fix_data_dir.sh data/${trainset}_u${datasize}k_per_ch/CH1
        for CH in 2 3 4; do
            utils/copy_data_dir.sh data/${trainset}_uall data/${trainset}_u${datasize}k_per_ch/CH${CH}
            sed -e "s/.CH1-/.CH${CH}-/g" data/${trainset}_u${datasize}k_per_ch/CH1/text > data/${trainset}_u${datasize}k_per_ch/CH${CH}/text
            utils/fix_data_dir.sh data/${trainset}_u${datasize}k_per_ch/CH${CH}
        done
        utils/combine_data.sh data/${trainset}_u${datasize}k \
            data/${trainset}_u${datasize}k_per_ch/CH1 \
            data/${trainset}_u${datasize}k_per_ch/CH2 \
            data/${trainset}_u${datasize}k_per_ch/CH3 \
            data/${trainset}_u${datasize}k_per_ch/CH4
        
        rm -rf data/${trainset}_u${datasize}k_per_ch
    done

    # use ch L to split
    for trainset in ${trainsets}; do
        utils/copy_data_dir.sh data/${trainset}_worn data/${trainset}_worn_u${worn_size}k_per_ch/L
        grep "\.L-" data/${trainset}_worn/text > data/${trainset}_worn_u${worn_size}k_per_ch/L/text
        echo "subset file"
        local/subset.py --text data/${trainset}_worn_u${worn_size}k_per_ch/L/text --outfolder data/${trainset}_worn_u${worn_size}k_per_ch/L --utters $((${worn_size} * 1000))
        utils/fix_data_dir.sh data/${trainset}_worn_u${worn_size}k_per_ch/L
        for CH in R; do
            utils/copy_data_dir.sh data/${trainset}_worn data/${trainset}_worn_u${worn_size}k_per_ch/${CH}
            sed -e "s/.L-/.${CH}-/g" data/${trainset}_worn_u${worn_size}k_per_ch/L/text > data/${trainset}_worn_u${worn_size}k_per_ch/${CH}/text
            utils/fix_data_dir.sh data/${trainset}_worn_u${worn_size}k_per_ch/${CH}
        done

        utils/combine_data.sh data/${trainset}_worn_u${worn_size}k data/${trainset}_worn_u${worn_size}k_per_ch/L data/${trainset}_worn_u${worn_size}k_per_ch/R  
        rm -rf data/${trainset}_worn_u${worn_size}k_per_ch
    done

fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    folders=""
    for x in u${datasize}k worn_u${worn_size}k ; do
        for y in ${trainsets}; do
            folders="${y}_${x} ${folders}"
        done
    done
    for x in ${folders} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    folders=""
    for x in u${datasize}k worn_u${worn_size}k ; do
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
if [ ${stage} -le 3 ]; then
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
    for x in u${datasize}k worn_u${worn_size}k ; do
        for y in ${trainsets}; do
            folders="${y}_${x} ${folders}"
        done
    done
    for rtask in ${folders}; do
        feat_tr=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_tr}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_tr}
    done

    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

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
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 4 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/train_worn_uall/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
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
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_mode${emode}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
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

if [ ${stage} -le 5 ]; then
    if [ "${emode}" == "regular" ]; then
        cp ${dumpdir}/train_u${datasize}k/delta${do_delta}/data.json ${feat_tr_dir}/data.json
    else
        dfiles=""
        for dset in ${trainsets}; do
            dfiles="${dumpdir}/${dset}_worn_u${worn_size}k/delta${do_delta}/data.json ${dfiles}"
        done
        for dset in train; do
            dfiles="${dumpdir}/${dset}_u${datasize}k/delta${do_delta}/data.json ${dfiles}"
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
        --einputs ${einputs} \
        --minput ${emode} \
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

if [ ${stage} -le 6 ]; then
    echo "stage 5: Decoding"
    nj=16

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
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
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


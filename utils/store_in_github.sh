#!/bin/bash

[ -f ./path.sh ] && . ./path.sh

DOWNLOAD_DIR=download

. utils/parse_options.sh || exit 1;


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ROOT_DIR=${PWD}

forced_read(){
    # <> {var name} {output text}
    forced_var=""
    while [ ! -n "${forced_var}" ] || [ ! -e ${forced_var} ]; do
        echo "[Required] $1"
        read -e  forced_var
    done
}

echo "This script will upload the pretrained model to store in a Git Repository"
echo "Insert Language model (ex: exp/train_rnnlm/rnnlm.model.best):"
read -e lm
if [ ! -n "${lm}" ]; then
    echo "Language model empty, skipping."
fi

forced_read "Insert Dictionary folder (ex: data/lang_char):"
dict=${forced_var}

forced_read "Insert Training config (ex: conf/train.yaml):"
tr_conf=${forced_var}

forced_read "Insert Decoding config (ex: conf/decode.yaml):"
dec_conf=${forced_var}

echo "Insert CMVN file (ex: data/tr_it/cmvn.ark):"
read -e cmvn
if [ ! -n "${cmvn}" ]; then
    echo "CMVN file empty, skipping."
fi

echo "Insert preprocessing file (ex: conf/preprocessing.yaml):"
read -e preprocess_conf
if [ ! -n "${preprocess_conf}" ]; then
    echo "Preprocessing file empty, skipping."
fi

forced_read "Insert Trained model (ex: exp/tr_it_pytorch_train/results/model.last10.avg.best):"
e2e=${forced_var}

echo "Insert the git of the model:"
read git_repo

mkdir -p ${DOWNLOAD_DIR}
cd ${DOWNLOAD_DIR}

repo_name=$(basename ${git_repo})
if [ -d ${repo_name} ]; then
    echo "Directory exists. Using that directory."
    echo "WARNING: If you want to download again the repository, 
          Delete the directory and run this script again. "
else
    echo "Downloading in: ${DOWNLOAD_DIR}/${repo_name}"
    git clone ${git_repo}
fi

echo "Files will be moved and stored into the git repo.
      Do you wish to continue? (Press Enter or [Ctrl + C] to escape.)"
read novar

cd ${ROOT_DIR}

save_file(){
    my_dir=$(dirname $1)
    save_dir=${DOWNLOAD_DIR}/${repo_name}/${my_dir}
    mkdir -p ${save_dir}
    cp $1 ${DOWNLOAD_DIR}/${repo_name}/$1
}

save_model(){
    my_dir=$(dirname $1)
    save_dir=${DOWNLOAD_DIR}/${repo_name}/${my_dir}
    mkdir -p ${save_dir}
    tar -cvzf - $1 | split -b 49M - ${DOWNLOAD_DIR}/${repo_name}/$1.tar.gz.
}

if [ -n "${lm}" ]; then
    if [ -e ${lm} ]; then
        save_model ${lm}
        lm_conf=$(dirname ${lm})/model.json
        save_file ${lm_conf}
    fi
fi
if [ -n "${cmvn}" ]; then
    if [ -e ${cmvn} ]; then
        save_file ${cmvn}
    fi
fi

save_file ${tr_conf}
save_file ${dec_conf}

if [ -n "${preprocess_conf}" ]; then
    if [ -e ${preprocess_conf} ]; then
       save_file ${preprocess_conf}
    fi
fi

save_model ${e2e}
e2e_conf=$(dirname ${e2e})/model.json
save_file ${e2e_conf}

adict=${DOWNLOAD_DIR}/${repo_name}/${dict}
mkdir -p ${adict}
cp -R ${dict} ${adict}

cd ${DOWNLOAD_DIR}/${repo_name}

rm -rf .git
git init
git add .
git commit -m "Add pre-trained model file"
git remote add origin ${git_repo}
git push -u --force origin master

exit 0

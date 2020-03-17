#!/bin/bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <datadir> <dtype> <dst-dir>"
  echo "e.g.: $0 /export/corpus/data/fearless Dev data/Dev"
  exit 1
fi

datadir=$1
dtype=$2
dst=$3

audio=${datadir}/Audio/Segments/ASR_track2/${dtype}
transcripts=${datadir}/Transcripts/ASR_track2/${dtype}
# all utterances are WAV

spk_file=${transcripts}/FS02_ASR_track2_uttID2spkID_${dtype}

mkdir -p $dst || exit 1

[ ! -d $datadir ] && echo "$0: no such directory $datadir" && exit 1
if [ ! "${dtype}" == "Eval" ]; then
  [ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1
fi

dst=`pwd`/${dst}
wav_scp=${dst}/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=${dst}/text; [[ -f "$trans" ]] && rm $trans
utt2spk=${dst}/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2utt=${dst}/spk2utt; [[ -f "$spk2utt" ]] && rm $spk2utt

utils=`pwd`/utils
tmpdir=$(mktemp -d ${dst}/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT
cd ${tmpdir}
find ${audio} -iname '*.wav' | sort -k1,1 > ${dtype,,}.flist

#make a transcription
if [ ! "${dtype}" == "Eval" ]; then
  cat ${spk_file} | tr '[a-z]' '[A-Z]' | awk '{print $2"-"$1,"\t",$2}' > ${dtype,,}.utt2spk.tmp
  cat ${dtype,,}.utt2spk.tmp | awk '{print $1}' > ${dtype,,}.ids 
  cat ${dtype,,}.utt2spk.tmp | sort -k 1 > ${dtype,,}.utt2spk
  # echo " " >> ${dtype,,}.utt2spk

  paste -d" " ${dtype,,}.ids ${dtype,,}.flist | sort -k 1 > ${dtype,,}_wav.scp

  cut -f 2- -d" " ${transcripts}/FS02_ASR_track2_transcriptions_${dtype} | tr '[a-z]' '[A-Z]' > ${dtype,,}.words
  paste -d" " ${dtype,,}.ids ${dtype,,}.words | sort -k 1 > ${dtype,,}.txt

  cat ${dtype,,}.utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${dtype,,}.spk2utt || exit 1;
  opts=""
else
  cat ${dtype,,}.flist | awk -F'[/]' '{print $NF}' | sed -e 's/\.wav//' | tr '[a-z]' '[A-Z]' > ${dtype,,}_wav.ids 
  paste -d" " ${dtype,,}_wav.ids ${dtype,,}.flist | sort -k 1 > ${dtype,,}_wav.scp

  # Currently there is no transcripts for eval
  paste -d" " ${dtype,,}_wav.ids  | sort -k 1 | awk '{printf("%s [UNK]\n", $1);}' > ${dtype,,}.txt
  cat ${dtype,,}_wav.ids | awk '{printf("%s UNK\n", $1);}' | sort -k 1 > ${dtype,,}.utt2spk
  # echo " " >> ${dtype,,}.utt2spk
  cat ${dtype,,}.utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${dtype,,}.spk2utt || exit 1;
  opts="--no-text"
fi

# copying data to data/...
cp ${dtype,,}_wav.scp ${wav_scp} || exit 1;
cp ${dtype,,}.txt     ${trans}   || exit 1;
cp ${dtype,,}.spk2utt ${spk2utt} || exit 1;
cp ${dtype,,}.utt2spk ${utt2spk} || exit 1;

cd ${dst}/../..
${utils}/validate_data_dir.sh --no-feats --no-spk-sort ${opts} ${dst} || exit 1

echo "$0: successfully prepared data in ${dst}"

exit 0

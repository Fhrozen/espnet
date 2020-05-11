#!/bin/bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
# Apache 2.0

# Begin configuration section.
# End configuration section
is_stream=0

. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <json-transcript-dir> <output-dir>"
  echo -e >&2 "eg:\n  $0 /corpora/chime5/audio/train /corpora/chime5/transcriptions/train data/train"
  exit 1
fi

set -e -o pipefail

datadir=$1
dtype=$2

adir=${datadir}/Audio/Streams/${dtype}
jdir=${datadir}/Transcripts/ASR_track1/${dtype}

dir=$3


wav_count=$(find $adir -name "*.wav" | wc -l)
if [ "$wav_count" -eq 0 ]; then
  echo >&2 "We expect that the directory $adir will contain wav files."
  echo >&2 "That implies you have supplied a wrong path to the data."
  exit 1
fi

if [ ! "${dtype}" == "Eval" ] && [ ${is_stream} -eq 0 ] ; then
  json_count=$(find $jdir -name "*.json" | wc -l)
  if [ "$json_count" -eq 0 ]; then
    echo >&2 "We expect that the directory $jdir will contain json files."
    echo >&2 "That implies you have supplied a wrong path to the data."
    exit 1
  fi
  echo "$0: Converting transcription to text"

  mkdir -p $dir
  for file in $jdir/*json; do
    ./local/json2text.py $file
  done | \
    sed -e "s/\[inaudible[- 0-9]*\]/[inaudible]/g" |\
    sed -e 's/ - / /g' |\
    sed -e 's/mm-/mm/g' > $dir/text.orig

  echo "$0: Creating datadir $dir for type=\"$dtype\""

  # fixed reference

  # first get a text, which will be used to extract reference arrays
  cat $dir/text.orig | sort > $dir/text
  find $adir -name "*.wav" |\
    perl -ne '$p=$_;chomp $_;@F=split "/";$F[$#F]=~s/\.wav//;$G= uc $F[$#F];print "$G $p";' |\
    sort -u  > $dir/wav.scp

  $cleanup && rm -f $dir/text.* $dir/wav.scp.* $dir/wav.flist

  # Prepare 'segments', 'utt2spk', 'spk2utt'
  cut -d" " -f 1 $dir/text | \
      awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $2, $3/100.0, $4/100.0)}' > $dir/segments 
      # awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $1, $2/100.0, $3/100.0)}' |\
      # sed -e "s/_[A-Z]*\././2" |\
      # sed -e 's/ P.._/ /' > $dir/segments

  cut -f 1 -d ' ' $dir/segments | \
    perl -ne 'chomp;$utt=$_;s/-.*//;print "$utt $_\n";' > $dir/utt2spk

  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
else
  mkdir -p $dir
  
  find $adir -name "*.wav" |\
  perl -ne '$p=$_;chomp $_;@F=split "/";$F[$#F]=~s/\.wav//;$G= uc $F[$#F];print "$G $p";' |\
  sort -u  > $dir/wav.scp
  echo " " > $dir/text
  cut -f 1 -d ' ' $dir/wav.scp | tr "\n" " " | awk '{print "UNK\t" $0 }' >  $dir/spk2utt
  cut -f 1 -d ' ' $dir/wav.scp | awk '{print $0 "\t[unk]"}' >  $dir/text
  cut -f 1 -d ' ' $dir/wav.scp | awk '{print $0 "\tUNK"}' >  $dir/utt2spk
fi

# Check that data dirs are okay!
utils/validate_data_dir.sh --no-feats $dir || exit 1

#!/bin/bash

audio_dir=$1


tr_simu_list=$(find ${audio_dir}  -name '*CH1.wav' | grep 'tr05_bus_simu\|tr05_caf_simu\|tr05_ped_simu\|tr05_str_simu' | sort -u)
tr_real_list=$(find ${audio_dir} -name '*CH1.wav' | grep 'tr05_bus_real\|tr05_caf_real\|tr05_ped_real\|tr05_str_real' | sort -u)

for item in ${tr_simu_list} ${tr_real_list}; do
    channel1=$(perl -le 'print 1+int(rand(6))')
    channel2=$(perl -le 'print 1+int(rand(6))')
    while [ ${channel1} -eq ${channel2} ]; do
        channel2=$(perl -le 'print 1+int(rand(6))')
    done
    in_ch1=${item//CH1/CH$channel1}
    in_ch2=${item//CH1/CH$channel2}
    out_ch1=${in_ch1//isolated/isolated_2ch_track}
    out_ch2=${in_ch2//isolated/isolated_2ch_track}

    output_dir=$(dirname "${out_ch1}")
    mkdir -p ${output_dir}
    cp ${in_ch1} ${out_ch1}
    cp ${in_ch2} ${out_ch2}
done

echo "Finished"
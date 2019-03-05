#!/bin/bash

list=$( find ./dump -iname "data.json" )

for i in ${list};do
    if [ ! -z ${i}.old ]; then
        cp ${i} ${i}.old
    fi
    echo ${i} 
    sed -i 's|/espnet/egs/ami/mc_asr1|.|g' ${i}
    
done

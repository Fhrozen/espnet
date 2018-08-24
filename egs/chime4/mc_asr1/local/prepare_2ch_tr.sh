#!/bin/bash

datadir=$1


find ${datadir} 

for item in ${list}; do
    channel1=$(perl -le 'print 1+int(rand(6))')
    channel2=$(perl -le 'print 1+int(rand(6))')
    while [ ${channel1} -eq ${channel2} ]; do
        channel2=$(perl -le 'print 1+int(rand(6))')
    done
    
done

echo "Finished"
#!/bin/bash

expdir=exp/train_chainer_transf_wave_res_large_feats_wave/decode_*_model.val5.avg.best_ctc0.1/hyp.trn

./local/prepare_submission.py \
    --results ${expdir} 
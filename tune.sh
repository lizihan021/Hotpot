#! /bin/bash
python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
        "dropout=[1/2, 2/3, 3/4]" "inp_e_dropout=[1/2, 3/4, 4/5]" "l2reg=[1e-4, 1e-3, 1e-2]"

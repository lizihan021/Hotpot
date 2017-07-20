#! /bin/bash
python tools/train.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv inp_e_dropout=1/2 epochs=64

#! /bin/bash
python tools/eval.py avg sts data/all/2015.train.tsv \
data/all/2015.train.tsv data/all/2015.val.tsv weights-sts-avg--dea53298481760-* inp_e_dropout=1/2 epochs=64

#! /bin/bash
python tools/train.py avg sts datasets/all/2015.train.tsv datasets/all/2015.val.tsv inp_e_dropout=1/2 nb_epoch=64

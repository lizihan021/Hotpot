#! /bin/bash
python3 tools/train.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv

#python3 tools/train.py avg para data/question-pairs-dataset/train.csv \
#data/question-pairs-dataset/test.csv batch_size=1

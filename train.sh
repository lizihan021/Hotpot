#! /bin/bash
python3 tools/train.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv

#python3 tools/train.py gran para data/question-pairs-dataset/train-small.csv \
#data/question-pairs-dataset/test.csv

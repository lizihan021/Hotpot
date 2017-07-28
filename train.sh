#! /bin/bash
#python3 tools/train.py dnn sts data/all/2015.train.tsv \
#data/all/2015.val.tsv

python3 tools/train.py avg para data/question-pairs-dataset/questions.csv \
data/question-pairs-dataset/questions.csv

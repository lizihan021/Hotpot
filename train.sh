#! /bin/bash
python3 tools/train.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv embprune=200 

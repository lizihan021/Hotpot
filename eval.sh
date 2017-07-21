#! /bin/bash
# format: MODEL TASK TRAIN_FILE VAL_FILE TRAIN_FILE(- for none) WEIGHT_FILE ARGS 
python3 tools/eval.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv - weights-sts-avg--dea53298481760-* 

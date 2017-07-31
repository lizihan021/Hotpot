#! /bin/bash
# format: MODEL TASK TRAIN_FILE VAL_FILE TRAIN_FILE(- for none) WEIGHT_FILE ARGS 
python3 tools/eval.py avg para data/question-pairs-dataset/train.csv \
data/question-pairs-dataset/test.csv - weights/weights-para-avg--* batch_size=1

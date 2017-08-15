#! /bin/bash
# format: MODEL TASK TRAIN_FILE VAL_FILE TEST_FILE(- for none) WEIGHT_FILE ARGS 
python3 tools/eval.py gran para data/question-pairs-dataset/train.csv \
data/question-pairs-dataset/test.csv - weights/weights-para-gran--6* 

#! /bin/bash
<<<<<<< HEAD
python3 tools/train.py avg sts data/all/2015.train.tsv \
data/all/2015.val.tsv 
=======
#python3 tools/train.py dnn sts data/all/2015.train.tsv \
#data/all/2015.val.tsv

python3 tools/train.py avg para data/question-pairs-dataset/questions.csv \
data/question-pairs-dataset/questions.csv
>>>>>>> 90686f450e5bc2779fc53ce6edee20697a2bc4c6

#! /bin/bash
python3 tools/tuning.py cnn anssel data/anssel/wang/train-all.csv data/anssel/wang/dev.csv \
        "dropout=[1/2, 2/3, 3/4]" "inp_e_dropout=[1/2, 3/4, 4/5]" "l2reg=[1e-4, 1e-3, 1e-2]" \
        "project=[True, True, False]" "cnnact=['tanh', 'relu']" \
        "cdim={1: [0,0,1/2,1,2], 2: [0,0,1/2,1,2,0], 3: [0,0,1/2,1,2,0], 4: [0,0,1/2,1,2,0], 5: [0,0,1/2,1,2]},"

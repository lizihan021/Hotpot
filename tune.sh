#! /bin/bash
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		embprune=200
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		embprune=100
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		embicase=True
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		embicase=False
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		"e_add_flags=[True, False]" "inp_e_dropout=[1/3, 1/2, 0.65, 0.8]" \
# 		"inp_w_dropout=[1/3, 1/2]"
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		"mlpsum=['sum', 'absdiff']" "Ddim=[1, 2]"
# python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
# 		"batch_size=[80, 160, 320]"
python3 tools/tuning.py avg sts data/all/2015.train.tsv data/all/2015.val.tsv \
		"l2reg=[1e-5, 1e-4, 1e-3, 1e-2]" "wproject=[True, False]" "deep=[1]" \
		"project=[True, False]"
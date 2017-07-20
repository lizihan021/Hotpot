The Ubuntu Dialogue Corpus
==========================

This is not an "answer sentence selection" problem per se, but it is the same
kind of bipartite ranking task.  It was introduced in [(Lowe, 1506.08909)](http://arxiv.org/abs/1506.08909).

There are two datasets.  v1, which has two published papers comparing models
with it, and v2, which is a nicer dataset but the results are unpublished yet.
We champion v2, also for technical reasons.

This corpus is obviously too big to be included as-is in this repository.
To get it, follow per-version instructions below.

Regarding software, use ``tools/ubuntu_preprocess.py`` and ``tools/train.py``
(on ``ubuntu`` task).
See instructions on top of ``tasks/ubuntu.py`` re preprocessing
the csv files (the dataset is too large to be fed directly to the
train tool).

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.

Note that NO DROPOUT is applied for any of the models.  In case of DAN, that
also means inp_w_dropout=0.

Also NOTE THAT THESE RESULTS ARE OBSOLETE because they predate the f/bigvocab port.

| Model                    | valMRR   | val-2R@1 | val-10R@2 | testMRR  | test-2R@1 | test-10R@2 | settings
|--------------------------|----------|----------|-----------|----------|-----------|------------|---------
| TF-IDF (Ryan Lowe)       |          |          |           |          | 0.749260  | 0.587315   | personal communication
| RNN (Ryan Lowe)          |          |          |           |          | 0.776539  | 0.560689   | personal communication
| LSTM (Ryan Lowe)         |          |          |           |          | 0.868731  | 0.720991   | personal communication
|--------------------------|----------|----------|-----------|----------|-----------|------------|---------
| avg                      | 0.619594 | 0.790865 | 0.603921  | 0.623905 | 0.793301  | 0.608480   |
|                          |±0.001845 |±0.002024 |±0.002699  |±0.001948 |±0.002199  |±0.002341   |
| DAN                      | 0.573181 | 0.788976 | 0.609164  | 0.577813 | 0.791596  | 0.614717   | ``deep=2`` ``pact='relu'``
|                          |±0.069417 |±0.035779 |±0.060425  |±0.069916 |±0.035020  |±0.059468   |
|--------------------------|----------|----------|-----------|----------|-----------|------------|---------
| rnn                      | 0.779721 | 0.905470 | 0.797897  | 0.780736 | 0.906812  | 0.799088   | ``sdim=1`` ``pdim=1`` ``ptscorer=B.dot_ptscorer``
|                          |±0.003547 |±0.002233 |±0.004258  |±0.003112 |±0.001660  |±0.003981   |
| cnn                      | 0.713658 | 0.859075 | 0.715114  | 0.717872 | 0.863471  | 0.720686   | ``pdim=1`` ``ptscorer=B.dot_ptscorer``
|                          |±0.003120 |±0.002882 |±0.002481  |±0.003490 |±0.001821  |±0.005233   |
| rnncnn                   | 0.786378 | 0.909691 | 0.804825  | 0.787976 | 0.910895  | 0.808549   | ``sdim=1/2`` ``pdim=1`` ``ptscorer=B.dot_ptscorer``
|                          |±0.001146 |±0.000928 |±0.001801  |±0.001286 |±0.000801  |±0.001530   |
| attn1511                 | 0.768170 | 0.898332 | 0.783237  | 0.771859 | 0.901619  | 0.787969   | ``sdim=1/2`` ``cdim=1/2`` ``ptscorer=B.dot_ptscorer``
|                          |±0.004161 |±0.003382 |±0.005648  |±0.003974 |±0.002407  |±0.005307   |

These results are obtained like this:

	tools/train.py avg ubuntu data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle "vocabf='data/anssel/ubuntu/v2-vocab.pickle'" nb_runs=16 inp_e_dropout=0 dropout=0
	tools/eval.py avg ubuntu data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle data/anssel/ubuntu/v2-testset.pickle weights-ubuntu-avg--69489c8dc3b6ce11-*-bestval.h5 "vocabf='data/anssel/ubuntu/v2-vocab.pickle'" inp_e_dropout=0 dropout=0


Corpus v2.0
-----------

To download v2 generated by the process below, use (in this (data/anssel/ubuntu/)
directory):

	wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-trainset.csv.gz
	wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-valset.csv.gz
	wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-testset.csv.gz
	gunzip v2*.gz

To generate v2, we first need to generate the sentence pairs, allow about 12
hours for this process.  Then, we need to post-process it for tokenization
(using brendan o'connor's tweet tokenizer, which deals right with "chatspeak":
URLs, emoticons etc.; we use a slightly tweaked version).

Run these commands in this (data/anssel/ubuntu/) directory:

	git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator
	cd ubuntu-ranking-dataset-creator/src
	./generate.sh
	cd ../../

	git clone https://github.com/brmson/tweetmotif
	touch tweetmotif/__init__.py
	./preprocess.py ubuntu-ranking-dataset-creator/src/train.csv v2-trainset.csv train
	./preprocess.py ubuntu-ranking-dataset-creator/src/valid.csv v2-valset.csv test
	./preprocess.py ubuntu-ranking-dataset-creator/src/test.csv v2-testset.csv test

There are no officially published reference results on the v2.0 corpus yet
(Feb 2016), but Ryan Lowe kindly shared the results of the originally published
baselines on the new corpus:

	TF-IDF:
	2-R@1: 0.749260042283
	10-R@1: 0.48810782241  10-R@2: 0.587315010571  10-R@5: 0.763054968288

	RNN:
	2-R@1: 0.776539210705
	10-R@1: 0.379139142954  10-R@2: 0.560689786585  10-R@5: 0.836350355691

	LSTM:
	2-R@1: 0.868730970907
	10-R@1: 0.552213717862  10-R@2: 0.72099120433  10-R@5: 0.924285351827


Corpus v1.0
-----------

  * http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/

To get it, run these commands in this (data/anssel/ubuntu/) directory:

	for i in aa ab ac ad ae; do wget http://cs.mcgill.ca/~npow1/data/ubuntu_dataset.tgz.$i; done
	cat ubuntu_dataset.tgz.a* | tar xz
	mv ubuntu_csvfiles/trainset.csv v1-trainset.csv
	mv ubuntu_csvfiles/valset.csv v1-valset.csv
	mv ubuntu_csvfiles/testset.csv v1-testset.csv

The main problem with v1 is that we can't use it as-is anyway as the trainset
is wrong!  It does not include the postprocessing (tokenization(!!!), NE
substitution) that val and test datasets do.  It should be possible to
reconstruct that using

	https://github.com/ryan-lowe/Ubuntu-Dialogue-Generationv2
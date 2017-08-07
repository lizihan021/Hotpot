KeraSTS Training and Evaluation Tools
=====================================

Along with the datasets, we ship a variety of models that implement the *f_2*
text similarity functions.  These models should be task-independent, and they
can be run and applied on a specific task (family of datasets) using the tools
below.

**Getting Started:**
Basically, you pick a model from the **models/** directory, then pass its
name, dataset paths and possibly some parameter tweaks to the training tool
to train the model on that task/dataset and show validation set performance.

**Legacy:**
In the past, each task had a separate suite of scripts.  We are currently
converting the tasks to the generic interface, but meanwhile their scripts
are listed in sections below.

Training
--------

The basic thing you are going to need is to train a model on a task --- using
a training split of a dataset and evaluating it on the validation split.

Usage: ``tools/train.py MODEL TASK TRAINDATA VALDATA [PARAM=VALUE]...``

Example: ``tools/train.py gran para data/question-pairs-dataset/train.csv data/question-pairs-dataset/test.csv l2reg=0.0001 batch_size=32``

Evaluation
----------

The training script above creates a file with weights dump for a particular
trained model instance (or instances, with ``nb_runs`` parameter).  You can
use a set of instances to evaluate performance on datasets, statistically.

Usage: ``tools/eval.py MODEL TASK VOCABDATA TRAINDATA VALDATA TESTDATA WEIGHTFILES... [PARAM=VALUE]...``

Example: ``tools/eval.py gran para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv - weights-para-gran--254fd35feb13ceef-*``

(Instead of -, you can pass a test set for the very final evaluation.)

A custom dataset for constructing the vocabulary can be passed as a config
argument like:

	"vocabf='data/question-pairs-dataset/train.csv'"

(This is useful for evaluating a model on a different dataset than how it
was trained, or for datasets with external vocabulary, like ubuntu.)

Tuning
------

It may be useful to run a series of model trainings with different
hyperparmeter values, deriving the optimal configuration from evaluation
on the validation set.

Usage: ``tools/tuning.py MODEL TASK TRAINDATA VALDATA PARAM=VALUESET...``

For an example and details, see the header of the script.

Transfer Learning
-----------------

We have a gadget that loads a model trained on one task and retrains it
on another.

Usage: ``tools/transfer.py MODEL TASK1 VOCAB1 WEIGHTS TASK2 TRAIN2DATA VAL2DATA [PARAM=VALUE]...``

For an example and details, see the header of the script.

Visualization
-------------

We provide several IPython notebooks that use HTML for visualization of
attention and other aspects of the RNN.

**Warning:** These scripts are now slightly obsolete in that they rely
on the old task-specific tools instead of using the task-generic interface.
Some (generally pretty light?) changes are required to make them run;
please contribute them back. :)

  * **anssel-visual.ipynb** showcases IPython/Jupyter notebook for model
    development and visualizing attention models using a token heatmap

  * **ubuntu-visual.ipynb** is a simple transposition of anssel-visual for
    attention visualization on the Ubuntu corpus  (the Ubuntu dialogue
    corpus is the same task, but uses dedicated scripts to be
    able to process large datasets).

REST API
--------

  * **scoring-api.py** is a RESTful API for scoring provided sentence pairs
    based on a given model.
    For an example and details, see the header of the script.  This is a work
    in progress, the API may not be stable.  anssel-specific at the moment.

  * **retrieval-api.py** is analogous REST API but accepts just a single
    sentence, matching a hard-coded dataset of *s1* sentences and returning
    the best *k* matches.  para-specific at the moment.

  * **hypev-api.py** is also analogous, but specific to the hypev task.

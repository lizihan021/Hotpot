#!/usr/bin/python3
"""
Copyright 2017 Liang Qiu, Zihan Li, Yuanyi Ding

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Train a KeraSTS model on the given task, save the trained model
to a weights file and evaluate it on a validation set.

Usage: python3 tools/train.py MODEL TASK TRAINDATA VALDATA [PARAM=VALUE]...

Example: python3 tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2

Parameters are mostly task-specific and model-specific, see the
respective config() routines.  The training process itslef is
influenced by:

    * batch_size=N denotes number of samples per batch

    * epochs=N denotes maximum number of epochs (tasks will
      typically include a val-dependent early stopping mechanism too)

    * nb_runs=N denotes number of re-trainings to attempt (1 by default);
      final weights are stored for each re-training, this is useful to
      control for randomness-induced evaluation instability (see also
      tools/eval.py); it's very much like just running this script
      N times, except faster (no embedding and dataset reloading)

Some other noteworthy task,model-generic parameters (even if
not train-specific) are:

    * adapt_ubuntu=True to add __eot__ __eos__ tokens to the dataset sentences
      like they are in the Ubuntu Dialogue dataset (useful for transfer
      learning but also potentially for the models as markers)

    * f_add=[INPUT...] to add extra graph inputs to the final ptscorer classifier
      as additional features.  For example, some of the anssel datasets accept
      ``f_add=['kw', 'akw']`` to include prescoring keyword matching.

    * prescoring=MODEL, prescoring_conf={CONFDICT}, prescoring_weightsf=FILE
      to apply a pre-scoring step on the dataset using the given model
      with the given config, loaded from given file; the precise usage
      (whether as a feature, rank-based pruning, etc.) of prescoring is
      task-specific; Ex.:

        "prescoring='termfreq'" "prescoring_conf={freq_mode: 'tf'}" \
                "prescoring_weightsf='weights-anssel-termfreq--120a2d2e6dcd0c16-bestval.h5'"
                
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
import sys

import pysts.embedding as emb
from pysts.hyperparam import hash_params

import models  # importlib python3 compatibility requirement
import tasks

# TODO Unused imports for evaluating commandline params
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import *
from keras.utils import plot_model
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504
import pysts.kerasts.blocks as B
from tasks import default_config


# return the config combined by model_config and task_config
def config(model_config, task_config, params):
    c = default_config(model_config, task_config) # by default, model_config > task_config

    for p in params:
        keyword, value = p.split('=')
        c[keyword] = eval(value)

    ps, h = hash_params(c)

    return c, ps, h


def train_model(runid, model, task, c):
    print('Training')
    fit_kwargs = dict()
    if c['balance_class']: # TODO?
        one_ratio = np.sum(task.gr['score'] == 1) / len(task.gr['score'])
        fit_kwargs['class_weight'] = {'score': {0: one_ratio, 1: 0.5}}
    if 'score' in task.gr:
        n_samples = len(task.gr['score'])
    else:
        n_samples = len(task.gr['classes'])
    fit_kwargs['steps_per_epoch'] = int(n_samples * c['epoch_fract'])
    task.fit_model(model, weightsf='weights/weights-'+runid+'-bestval.h5',
                   batch_size=c['batch_size'], epochs=c['epochs'],
                   **fit_kwargs)
    # model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('weights/weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('weights/weights-'+runid+'-bestval.h5')

# used to build model and then call train_model
def train_and_eval(runid, module_prep_model, task, conf, do_eval=True):
    print('Model')
    model = task.build_model(module_prep_model)
    plot_model(model, to_file='model.png')
    
    train_model(runid, model, task, conf)

    if do_eval:
        print('Predict&Eval (best val epoch)')
        res = task.eval(model)
    else:
        res = None
    return model, res


if __name__ == "__main__":
    modelname, taskname, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    model_module = importlib.import_module('.'+modelname, 'models')
    task_module = importlib.import_module('.'+taskname, 'tasks')
    task = task_module.task()
    conf, ps, h = config(model_module.config, task.config, params)
    task.set_conf(conf)

    # configurable embedding class
    if conf['embdim'] is not None:
        print('GloVe')
        task.emb = emb.GloVe(N=conf['embdim'])
    else:
        task.emb = None

    print('Dataset')
    if 'vocabf' in conf: 
        task.load_vocab(conf['vocabf'])
    task.load_data(trainf, valf)
    print('Dataset loaded')

    for i_run in range(conf['nb_runs']):
        if conf['nb_runs'] == 1:
            runid = '%s-%s-%x' % (taskname, modelname, h)
        else:
            runid = '%s-%s-%x-%02d' % (taskname, modelname, h, i_run)
        print('RunID: %s  (%s)' % (runid, ps))

        train_and_eval(runid, model_module.prep_model, task, conf)

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


KeraSTS interface for datasets of the Paraphrasing task.  Basically,
it's a lot like STS but with binary output.  See data/para/... for
details and actual datasets.

Training example:
    tools/train.py cnn para data/para/msr/msr-para-train.tsv data/para/msr/msr-para-val.tsv inp_e_dropout=1/2
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

import pysts.eval as ev
from pysts.kerasts.callbacks import ParaCB
from pysts.kerasts import graph_input_anssel, graph_nparray_anssel
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


class ParaphrasingTask(AbstractTask):
    def __init__(self):
        self.name = 'para'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'binary_crossentropy'
        c['nb_epoch'] = 32

    def load_set(self, fname, lists=None):
        if lists:
            s0, s1, y = lists
        else:
#            s0, s1, y = loader.load_msrpara(fname)   #set it free is we decide not to use quora dataset
            s0, s1, y = loader.load_quora(fname)

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
        else:
            vocab = self.vocab

        si0, sj0 = vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        gr = graph_nparray_anssel(graph_input_anssel(si0, si1, sj0, sj1, None, None, y, f0, f1))

        return (gr, y, vocab)

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary')

        model = self.prep_model(module_prep_model)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss=self.c['loss'], optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [ParaCB(self, self.grv),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
                EarlyStopping(monitor='acc', mode='max', patience=3)]

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = self.predict(model, gr)
            tmp = []
            for yp in ypred:
                tmp.append(yp[0]) 
            ypred = np.array(tmp)
            res.append(ev.eval_para(ypred, gr['score'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f  |%s%.6f |%s%.6f |%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.trainf]['F1'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.valf]['F1'],
                  pfx, mres[self.testf].get('Accuracy', np.nan),
                  pfx, mres[self.testf].get('F1', np.nan)))


def task():
    return ParaphrasingTask()

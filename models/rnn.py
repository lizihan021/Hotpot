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


A model with a very simple architecture that never-the-less closely
approaches 2015-state-of-art results on the anssel-wang task (with
token flags).

The architecture uses shared bidirectional GRU to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe, projection
matrix (MemNN-like) applied to both sentences to project them to a common
external similarity space.

This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.
"""

from __future__ import print_function
from __future__ import division

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 4/5
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4

    c['rnnbidi'] = True
    c['rnn'] = GRU
    c['rnnbidi_mode'] = 'sum'
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2
    c['rnnlevels'] = 1

    c['project'] = True
    c['pdim'] = 2
    c['pact'] = 'tanh'
    c['pinit'] = 'glorot_uniform'

    # model-external:
    c['inp_e_dropout'] = 4/5
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2


def prep_model(inputs, N, s0pad, s1pad, c):
    outputs = B.rnn_input(inputs, N, s0pad,
                dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
                sdim=c['sdim'],
                rnnbidi=c['rnnbidi'], rnn=c['rnn'], rnnact=c['rnnact'], rnninit=c['rnninit'],
                rnnbidi_mode=c['rnnbidi_mode'], rnnlevels=c['rnnlevels'])

    # Projection
    if c['project']:
        proj = Dense(int(N*c['pdim']), activation=c['pact'], kernel_regularizer=l2(c['l2reg']), name='proj')
        e0p = proj(outputs[0])
        e1p = proj(outputs[1])
        N = N*c['pdim']
        return [e0p, e1p], N 
    else:
        return [outputs[0], outputs[1]], N

    #input_dim=int(N*c['sdim'])

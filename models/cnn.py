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

A model with a very simple architecture that never-the-less achieves
2015-state-of-art results on the anssel-wang task (with token flags).
You can also see this model as a standalone fully contained script
in ``examples/anssel-cnn.py``.

The architecture uses multi-width CNN and max-pooling to produce sentence embeddings,
adaptable word embedding matrix preinitialized with 300D GloVe and a projection
matrix (MemNN-like) applied to both sentences to project them to a common
external similarity space.

This will be a part of our upcoming paper; meanwhile, if you need to cite this,
refer to the dataset-sts GitHub repo, please.
"""

from __future__ import print_function
from __future__ import division

from keras.layers import Activation, Dense, Dropout
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['dropout'] = 0
    c['l2reg'] = 1e-4

    c['cnnsiamese'] = True
    c['cnnact'] = 'relu'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = {1: 1, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}

    c['project'] = False
    c['pdim'] = 2
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 0
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(inputs, N, s0pad, s1pad, c):
    Nc, outputs = B.cnnsum_input(inputs, N, s0pad, siamese=c['cnnsiamese'],
                        dropout=c['dropout'], l2reg=c['l2reg'],
                        cnninit=c['cnninit'], cnnact=c['cnnact'], cdim=c['cdim'])

    # Projection
    if c['project']:
        outputs = Dense(int(N*c['pdim']), kernal_regularizer=l2(c['l2reg']), activation=c['pact'])(outputs)
        # model.add_shared_node(name='proj', inputs=['e0s_', 'e1s_'], outputs=['e0p', 'e1p'],
        #                       layer=Dense(input_dim=Nc, output_dim=int(N*c['pdim']),
        #                                   W_regularizer=l2(c['l2reg']), activation=c['pact']))
        # This dropout is controversial; it might be harmful to apply,
        # or at least isn't a clear win.
        # model.add_shared_node(name='projdrop', inputs=['e0p', 'e1p'], outputs=['e0p_', 'e1p_'],
        #                       layer=Dropout(c['dropout'], input_shape=(N,)))
        # return ('e0p_', 'e1p_')
    return outputs, N

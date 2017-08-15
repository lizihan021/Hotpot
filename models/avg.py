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


A simple averaging model.

In its default settings, this is the baseline unigram (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

However, rather than a dot-product, the MLP comparison is used as it works
dramatically better.

This model can also represent the Deep Averaging Networks
(http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf) with this configuration:

    inp_e_dropout=0 inp_w_dropout=1/3 deep=2 "pact='relu'"

The model also supports preprojection of embeddings (not done by default;
wproj=True), though it doesn't do a lot of good it seems - the idea was to
allow mixin of NLP flags.


Performance:
    * anssel-yodaqa:
      valMRR=0.334864 (dot)
"""

from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Lambda
from keras import backend as K
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-4

    # word-level projection before averaging
    c['wproject'] = False
    c['wdim'] = 1
    c['wact'] = 'linear'

    c['deep'] = 0
    c['nnact'] = 'relu'
    c['nninit'] = 'glorot_uniform'

    c['project'] = True
    c['pdim'] = 1
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 1/3
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(inputs, N, s0pad, s1pad, c):
    # Word-level projection before averaging
    if c['wproject']:
        wproj = Dense(int(N*c['wdim']), activation=c['wact'], kernel_regularizer=l2(c['l2reg']), name='wproj')
        inputs[0] = wproj(inputs[0])
        inputs[1] = wproj(inputs[1])
    
    # Averaging
    avg = Lambda(function=lambda x: K.mean(x, axis=1), 
                 output_shape=lambda shape: (shape[0], ) + shape[2:])
    e0b = avg(inputs[0])
    e1b = avg(inputs[1])
    bow_last = [e0b, e1b]
    
    # Deep
    for i in range(c['deep']):
        deepD1 = Dense(N, activation=c['nnact'], kernel_regularizer=l2(c['l2reg']), name='deep_%d'%(i,))
        bow_next_0 = deepD1(bow_last[0])
        bow_next_1 = deepD1(bow_last[1])
        bow_last = [bow_next_0, bow_next_1]

    # Projection
    if c['project']:
        proj = Dense(int(N*c['pdim']), activation=c['pact'], kernel_regularizer=l2(c['l2reg']), name='proj')
        e0b = proj(bow_last[0])
        e1b = proj(bow_last[1])
        N = N*c['pdim']
        return [e0b, e1b], N 
    else:
        return bow_last, N

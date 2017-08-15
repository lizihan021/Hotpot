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


A simple dnn model that apply 4 dense layers on the word embeddings
"""

from __future__ import print_function
from __future__ import division

from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras import backend as K
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-5

    c['deep'] = 4
    c['nndim'] = 200
    c['nndropout'] = 0.1
    c['nnact'] = 'relu'
    c['nninit'] = 'glorot_uniform'

    # model-external:
    c['inp_e_dropout'] = 1/3
    c['inp_w_dropout'] = 0

    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(inputs, N, s0pad, s1pad, c):
    # Word-level projection before averaging
    inputs[0] = TimeDistributed(Dense(N, activation='relu'))(inputs[0])
    inputs[0] = Lambda(lambda x: K.max(x, axis=1), output_shape=(N, ))(inputs[0])
    inputs[1] = TimeDistributed(Dense(N, activation='relu'))(inputs[1])
    inputs[1] = Lambda(lambda x: K.max(x, axis=1), output_shape=(N, ))(inputs[1])
    merged = concatenate([inputs[0], inputs[1]])
    
    # Deep
    for i in range(c['deep']):
        merged = Dense(c['nndim'], activation=c['nnact'])(merged)
        merged = Dropout(c['nndropout'])(merged)
        merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)
    return [is_duplicate], N

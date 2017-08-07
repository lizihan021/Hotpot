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


A model that combines LSTM + AVG to produce sentence embeddings, adaptable 
word embedding matrix preinitialized with 300D GloVe. The LSTM layer takes 
word embedding sequences as input. The AVG layer takes the outputs sequences 
of LSTM layer as inputs.
"""
from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Lambda, LSTM
from keras import backend as K
from keras.regularizers import l2

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-5

    # model-external:
    c['inp_e_dropout'] = 1/3
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(inputs, N, s0pad, s1pad, c):
    # LSTM
    lstm = LSTM(N, return_sequences=True)
    lstm1 = lstm(inputs[0])
    lstm2 = lstm(inputs[1])
    
    # Averaging
    avg = Lambda(function=lambda x: K.mean(x, axis=1),
                 output_shape=lambda shape: (shape[0], ) + shape[2:])
    lstm_avg1 = avg(lstm1)
    lstm_avg2 = avg(lstm2)
    
    return [lstm_avg1, lstm_avg2], N
        

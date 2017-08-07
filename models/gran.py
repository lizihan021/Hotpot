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


A model with the 2017-state-of-art Gated Recurrent Averaging Network 
(GRAN) architecture that achieves a high prediction accuracy on the 
Quera Question Pairs competition task.

The architecture uses LSTM, averaging and a gating system to produce 
sentence embeddings, adaptable word embedding matrix preinitialized 
with 300D GloVe.

The key idea behind the GRAN is to add importance to different word 
embeddings prior to averaging. The gate can be seen as an attention, 
attending to the information of each step of the LSTM such that:

    a = x * sigmoid(W_x * x + W_h * h + b)

This model was inspired by John Weithing, a PhD student at the Toyota 
Technological Institute at the University of Chicago. Details of this 
model can be found at http://ttic.uchicago.edu/~wieting/wieting2017Recurrent.pdf
"""

from __future__ import print_function
from __future__ import division

from keras.layers import Dense, Lambda, LSTM, add, Activation, multiply
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


def prep_model(inputs, N, s0pad, s1pad, c, granlevels=1):
    # LSTM
    lstm = LSTM(N, return_sequences=True, implementation=2, 
                   kernel_regularizer=l2(c['l2reg']), recurrent_regularizer=l2(c['l2reg']),
                   bias_regularizer=l2(c['l2reg']))
    x1 = inputs[0]
    x2 = inputs[1]
    h1 = lstm(x1)
    h2 = lstm(x2)
     
    W_x = Dense(N, kernel_initializer='glorot_uniform', use_bias=True, 
                   kernel_regularizer=l2(c['l2reg']))
    W_h = Dense(N, kernel_initializer='orthogonal', use_bias=True,
                   kernel_regularizer=l2(c['l2reg']))
    sigmoid = Activation('sigmoid')
    a1 = multiply([x1, sigmoid( add([W_x(x1), W_h(h1)]) )])
    a2 = multiply([x2, sigmoid( add([W_x(x2), W_h(h2)]) )])
     
    # Averaging
    avg = Lambda(function=lambda x: K.mean(x, axis=1),
                 output_shape=lambda shape: (shape[0], ) + shape[2:])
    gran1 = avg(a1)
    gran2 = avg(a2)
    
    return [gran1, gran2], N
        

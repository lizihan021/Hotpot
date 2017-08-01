"""
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

from keras.layers import Dense, Lambda, LSTM, merge, add, Activation, TimeDistributed, multiply
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
    lstm_avg1 = avg(a1)
    lstm_avg2 = avg(a2)
    
    return [lstm_avg1, lstm_avg2], N
        

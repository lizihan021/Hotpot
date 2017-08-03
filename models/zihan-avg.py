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
    a1 = multiply([x1**3, sigmoid( add([W_x(x1), W_h(h1)]) )])
    a2 = multiply([x2**3, sigmoid( add([W_x(x2), W_h(h2)]) )])
     
    # Averaging
    avg = Lambda(function=lambda x: K.mean(x, axis=1),
                 output_shape=lambda shape: (shape[0], ) + shape[2:])
    gran1 = avg(a1)
    gran2 = avg(a2)
    
    return [gran1, gran2], N
        

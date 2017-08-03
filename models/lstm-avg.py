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
        

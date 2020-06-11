import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.layers import Layer
from keras import initializers

from keras.models import Sequential
from keras.models import Model
from batch import BatchGenerator,BatchGeneratorFeatureTemp
import tensorflow as tf
# from MeanOverTime import *
from keras.layers import Reshape,Bidirectional
from keras.layers import Input,merge,RepeatVector
from keras.layers import Masking,MaxPooling1D,Flatten,Permute,Lambda
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent,Convolution1D
from keras.layers import Activation,SimpleRNN
from keras.optimizers import Adam,SGD,rmsprop
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras import backend as K

import matplotlib
# matplotlib.use('Agg')
import theano
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure,ylim
from keras.preprocessing.sequence import pad_sequences


SEED=1337
tf.set_random_seed(SEED)
np.random.seed(SEED)
batch_size = 16
nb_epochs = 10
nb_classes =2
time_steps =41
in_file_length =10
data_version=5
padding_value =0
ma = Sequential()
ma.add(Dense(30, input_dim=20))
ma.add(Dense(15))

mb = Sequential()
mb.add(Dense(15,input_dim=50))

modelmerge = Sequential()
modelmerge.add(Merge(layers=[ma,mb], mode='concat'))
modelmerge.add(Dense(1))

modelmerge.compile(loss='mae',optimizer='sgd')

nb_train = 100
Xa = np.random.random((nb_train, 20))
print(Xa.shape)
Xb = np.random.random((nb_train, 50))
y_train = np.random.uniform(0,0.001, (nb_train, 1))



his = modelmerge.fit([Xa, Xb], y_train,batch_size=10,nb_epoch=2,)
print(his.history)

print([modelmerge.layers[0].layers[0].input,modelmerge.layers[0].layers[1].input])
get_left = K.function([modelmerge.layers[0].layers[0].input,modelmerge.layers[0].layers[1].input],
                      [modelmerge.layers[0].layers[0].output])
lout = get_left([Xa,Xb])
# print (lout.shape)
print (lout)
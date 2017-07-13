import numpy as np
import time
import csv
import random
from accessory import fun2,fun3,category_to_target,diff_length_csv,diff_length_dat,reshape_dataset
from temp_prediction_train_test import conv_lstm_model
from temp_prediction_train_test import model_model,basic_conv
from keras.models import Sequential
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers import Masking,MaxPooling1D,Flatten,Embedding
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent,Convolution1D
from keras.layers import Activation
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from accessory import get_train_test_data
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
import os
np.random.seed(1337)
batch_size = 16
nb_epochs = 200
nb_classes =2
temp_length=15
maxlen =50
in_file_length =87
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
is_conv_t_0_f_2 =0
data_version=1
superpara_a=0.1
input_predict_next_length=10
embedding_length =700

def diff_length_csv_10(filename):
    f_in = open(filename)
    return_list = []
    count=0
    for i, lines in enumerate(f_in):
        line = lines.strip().split(',')[0:]
        if len(line)>=10+1:
            return_list.append(line[0:11])
        else:
            count+=1
            return_list.append(line)
    print(count)
    return return_list

def joint_learning_model():

    input_temp = Input(shape=(maxlen - 1,), name='input_temp')
    reshapre_temp = Reshape(target_shape=(maxlen - 1, 1), name='reshape_temp')(input_temp)
    masking_temp = Masking(mask_value=is_conv_t_0_f_2)(reshapre_temp)
    lstm_temp = LSTM(16, name='lstm_temp')(masking_temp)
    dropout_lstm = Dropout(0.1)(lstm_temp)

    input_temp_next = Input(shape=(1,), name='input_temp_next')
    embedd_temp_next =Embedding(embedding_length,output_dim=16,)(input_temp_next)

    # for feature
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, name='dense_para')(input_para)
    dense_para = Dropout(0.1)(dense_para)
    dense_para = Activation('relu')(dense_para)

    merge_temp_para = merge([dense_temp, dropout_lstm, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.01),
                          b_regularizer=l2(0.01))(merge_temp_para)

    model = Model(input=[input_temp_next, input_temp, input_para], output=[dense_temp, dense_softmax])
    model.compile(loss={'dense_temp': 'mse', 'dense_softmax': 'categorical_crossentropy'},
                  optimizer=Adam(lr=0.001,clipnorm=1.),
                  metrics={'dense_temp': 'mse', 'dense_softmax': 'accuracy'})
    model.summary()

    return model

def joint_learning():

    X2 = fun2('number_age_col85tran_v2.csv')
    print(X2.shape)
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    X_train, X_test, y_train, y_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test,)

    temp_1_10 = fun2('front_1_10_temp.csv')
    temp_2_11 = fun2('front_2_11_temp.csv')

    temp_1_10_train,temp_1_10_test,temp_2_11_train,temp_2_11_test = get_train_test_data(temp_1_10,temp_2_11,nb_x_train,nb_x_test)
    temp_1_10_train = reshape_dataset(temp_1_10_train)
    temp_2_11_train = reshape_dataset(temp_2_11_train)
    temp_1_10_test = reshape_dataset(temp_1_10_test)
    temp_2_11_test = reshape_dataset(temp_2_11_test)
    print(temp_1_10_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length + 1:]
    x2_train = X_train[:, 0:in_file_length]

    x_test = X_test[:, in_file_length + 1:]
    x2_test = X_test[:, 0:in_file_length]

    print((sum(y_test) - len(y_test)) / len(y_test))
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model =joint_learning_model()
    mse_list=[]
    acc_list=[]
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit([temp_1_10_train,x_train,x2_train], [temp_2_11_train,y_train], batch_size=batch_size, nb_epoch=1,shuffle=True)
        loss,mse,acc = model.evaluate([temp_1_10_test,x_test,x2_test], [temp_2_11_test,y_test], batch_size=batch_size)
        print('loss:',loss)
        print('mse:',mse)
        print('acc:',acc)


if __name__ =='__main__':
    model =joint_learning_model()
    joint_learning()
    exit()

    # figure()
    for data_version in range(5,6):
        joint_learning()
    # show()

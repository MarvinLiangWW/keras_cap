import numpy as np
import time
import csv
import random
from accessory import fun2,fun3,category_to_target,diff_length_csv,diff_length_dat,reshape_dataset
from temp_prediction_train_test import conv_lstm_model
from temp_prediction_train_test import model_model,basic_conv
from keras.models import Sequential
from keras.models import Model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers import Masking,MaxPooling1D,Flatten,TimeDistributedDense
from keras.layers import LSTM, Dense,Merge,Dropout,SimpleRNN
from keras.layers import recurrent,Convolution1D
from keras.layers import Activation
from keras import optimizers
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from accessory import get_train_test_data
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
np.random.seed(1337)
batch_size = 16
nb_epochs = 200
nb_classes =2
temp_length=15
RNN = recurrent.LSTM
maxlen =50
in_file_length =87
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
is_conv_t_0_f_2 =0
data_version=1

def predict_model():
    input_temp =Input(shape=(input_temp_length,1),name='input_temp')
    rnn_temp=LSTM(output_dim=16,return_sequences=True,name='rnn_temp')(input_temp)
    dense_temp =Dense(1,name='dense_temp')(rnn_temp)
    dense_temp=Dropout(0.1)(dense_temp)
    dense_temp=Activation('relu')(dense_temp)

    model =Model(input=input_temp,output=dense_temp)
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='mse')
    model.summary()
    return model


def get_train_temp_instance(f_in ='2_5_day_tiwencheck.csv'):
    f_in =open(f_in)
    f_out=open('5_day_temp_longerthan_15.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        if len(line)>=16:
            f_out.write(lines)


def get_train_test_temp():
    f_in =open('5_day_temp_longerthan_15.csv')
    x_train=open('front_1_10_temp.csv','w')
    y_train =open('front_2_11_temp.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        x_train.write(line[0])
        y_train.write(line[0])
        for i in range(1,11):
            x_train.write(','+line[i])
            y_train.write(','+line[i+1])
        x_train.write('\n')
        y_train.write('\n')


def get_train_fun():
    x=open('front_1_10_temp.csv')
    y=open('front_2_11_temp.csv')
    x_train=[]
    y_train=[]
    for i,lines in enumerate(x):
        line=lines.strip().split(',')[1:]
        x_train.append(line)
    for i,lines in enumerate(y):
        line=lines.strip().split(',')[1:]
        y_train.append(line)
    return np.array(x_train),np.array(y_train)

#predict next temp
def baseline():
    x_array, y = get_train_fun()
    y=y[:,-1]
    # type changes from str to float
    y =[float(t) for t in y]
    x=[]
    for i in range(len(x_array)):
        result =[float(t) for t in x_array[i]]
        x.append(result)


    x_train = x[:400][:]
    y_train = y[:400]
    x_test = x[401:][ :]
    y_test = y[401:]
    lr = LinearRegression()
    lr = lr.fit(x_train, y_train, )
    r2 =r2_score(y_test, lr.predict(x_test))
    mse =mean_squared_error(y_test,lr.predict(x_test))
    print(r2)
    print('mse:',mse)

#predict next 10 temp
def baseline2():
    x_array, y_array = get_train_fun()
    x,y=[],[]
    for i in range(len(x_array)):
        x_result =[float(t) for t in x_array[i]]
        y_result =[float(t) for t in y_array[i]]
        x.append(x_result)
        y.append(y_result)
    x_train = x[:400][:][:]
    y_train = y[:400][:][:]
    x_test = x[401:][:][:]
    y_test = y[401:][:][:]
    print(x_test[0])
    print(y_test[0])
    lr = LinearRegression()
    lr = lr.fit(x_train, y_train, )
    print(lr.predict(x_test[0]))
    print(lr.predict(x_test).shape)
    mse = mean_squared_error(y_test, lr.predict(x_test))
    print('mse:', mse)

if __name__ =='__main__':
    baseline()
    # get_train_temp_instance('2_5_day_nor_s2.csv')
    # get_train_test_temp()

    input_temp_length = 10

    train_model = predict_model()
    x,y=get_train_fun()

    x=reshape_dataset(x)
    y=reshape_dataset(y)

    x_train =x[:400,:,:]
    y_train =y[:400,:,:]
    x_test =x[401:,:,:]
    y_test =y[401:,:,:]
    print(x_train.shape)


    mse_list=[]
    mse_train_list=[]
    mse =train_model.evaluate(x_test, y_test, batch_size=batch_size)
    mse_train =train_model.evaluate(x_train, y_train, batch_size=batch_size)
    mse_list.append(mse)
    mse_train_list.append(mse_train)
    mse_real_list =[]
    for epoch in range(nb_epochs):
        train_model.fit(x, y, verbose=1, batch_size=batch_size, nb_epoch=1, shuffle=True)
        mse_train =train_model.evaluate(x_train,y_train,batch_size=batch_size)
        mse = train_model.evaluate(x_test, y_test, batch_size=batch_size)
        mse_list.append(mse)
        mse_train_list.append(mse_train)
    plot(range(0, len(mse_list)), mse_list,label='mse_test')
    plot(range(0, len(mse_list)), mse_train_list,label='mse_train')
    print(mse_list)


    #predict 11-15
    for i in range(1,6):
        # test_model
        input_temp = Input(shape=(input_temp_length, 1), name='input_temp')
        rnn_temp = LSTM(output_dim=16, return_sequences=False, name='rnn_temp', )(input_temp)
        dense_temp = Dense(1, name='dense_temp')(rnn_temp)
        dense_temp = Activation('relu')(dense_temp)
        test_model = Model(input=input_temp, output=dense_temp)
        test_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                           loss='mse')
        weights = train_model.get_layer(name='rnn_temp').get_weights()
        test_model.get_layer('rnn_temp').set_weights(weights=weights)
        weight_dense = train_model.get_layer(name='dense_temp').get_weights()
        test_model.get_layer('dense_temp').set_weights(weights=weight_dense)

        x_test_temp = test_model.predict(x_test)
        # print(x_test_temp[0])
        x_temp = []
        for i in range(0, len(x_test_temp)):
            x_temp.append(np.append(x_test[i], x_test_temp[i][-1]))
        x_test = np.array(x_temp)
        print(x_test[0])
        x_test=reshape_dataset(x_test)
        input_temp_length = input_temp_length + 1
    legend()
    show()
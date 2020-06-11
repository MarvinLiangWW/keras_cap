from keras.models import Model
from keras.models import Sequential
import numpy as np
import keras.backend as K
from keras import layers
import keras
from keras.engine import Layer
from keras.datasets import mnist
import csv
from datetime import datetime,timedelta
import random
import time
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers import Masking,Embedding
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import Convolution2D,Convolution3D
from keras.layers import LSTM, Dense,Dropout,MaxPooling1D,MaxPooling2D,TimeDistributed
from keras.layers import Conv2D,Activation,Flatten,SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.optimizers import Adam,RMSprop,SGD
np.random.seed(1345)
batch_size = 16
epochs = 50
nb_classes =3
maxlen =20
in_file_length =73
max_med_len=10
learning_rate=0.01
len_medicine_list =317    #plus 1 for no medicine used
mid_dim =128

# K.set_image_data_format('channels_first')
class MeanoverTime(Layer):
    def __init__(self,**kwargs):
        self.supports_masking =True
        super(MeanoverTime,self).__init__(**kwargs)
    def call(self,x,mask=None):
        if mask is not None:
            mask = K.cast(mask,'float32')
            s =mask.sum(axis=2,keepdims =True)
            if K.equal(s,K.zeros_like(s)):
                return K.mean(x,axis=2)
            else :
                return K.cast(x.sum(axis=2)/mask.sum(axis =2,keepdims=True),K.floatx())
        else:
            return K.mean(x,axis=2)
    def get_output_shape_for(self,input_shape):
        # remove temporal dimension
        # return input_shape[0],input_shape[1],input_shape[3]
        return input_shape[0],input_shape[1],input_shape[-1]
    def compute_mask(self,input,input_mask=None):
        # do not pass the mask to the next layers
        return None


class MaxoverTime(Layer):
    def __init__(self,**kwargs):
        self.supports_masking =True
        super(MaxoverTime,self).__init__(**kwargs)

    def call(self,x,mask=None):
        if mask is not None:
            # mask = K.cast(mask,'float32')
            return K.max(x,axis=2)

        else:
            return K.max(x,axis=2)
    def get_output_shape_for(self,input_shape):
        # remove temporal dimension
        # return input_shape[0],input_shape[1],input_shape[3]
        return input_shape[0],input_shape[1],input_shape[-1]

    def compute_mask(self,input,input_mask=None):
        # do not pass the mask to the next layers
        return None

def read_split(filename):
    f_in =open(filename)
    label,number=[],[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number.append(line[0])
        label.append(line[1])
    return np.array(number),np.array(label)


def category_to_target(category):
    y =[]
    for i in range(0,category.shape[0]):
        temp=[]
        for k in range(0,nb_classes):
            if k+1==int(category[i]):
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
    return np.array(y,dtype=type(y[0][0]))


def preposed_model():
    input_med = Input(shape=(maxlen*max_med_len,), name='input_med')
    mask_med =Masking(mask_value=0)(input_med)
    em_med = Embedding(len_medicine_list + 1, output_dim=mid_dim, )(mask_med)
    dropout_med = Dropout(0.5)(em_med)
    reshape_med=Reshape((maxlen,max_med_len,mid_dim))(dropout_med)
    # [none,20,10,128]
    encoded_med = TimeDistributed(SimpleRNN(mid_dim),name ='TD1')(reshape_med)
    encoded_time = LSTM(mid_dim)(encoded_med)

    dense_softmax = Dense(nb_classes, name='dense_softmax', )(encoded_time)
    dense_softmax = Dropout(0.3)(dense_softmax)
    dense_softmax = Activation('softmax')(dense_softmax)
    model = Model(input=input_med, output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model

def sim_model():
    input_med = Input(shape=(maxlen * max_med_len,), name='input_med')
    mask_med = Masking(mask_value=0)(input_med)
    em_med = Embedding(len_medicine_list + 1, output_dim=mid_dim,)(mask_med)
    dropout_med = Dropout(0.5)(em_med)

    reshape_med_1 = Reshape((maxlen, max_med_len, mid_dim))(dropout_med)
    max_med = MaxoverTime()(reshape_med_1)

    lstm_med_temp = LSTM(mid_dim, )(max_med)

    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', )(lstm_med_temp)
    model = Model(input=input_med, output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()
    return model

def mean_model():
    input_med = Input(shape=(maxlen * max_med_len,), name='input_med')
    mask_med = Masking(mask_value=0)(input_med)
    em_med = Embedding(len_medicine_list + 1, output_dim=mid_dim,)(mask_med)
    dropout_med = Dropout(0.5)(em_med)

    reshape_med_1 = Reshape((maxlen, max_med_len, mid_dim))(dropout_med)
    max_med = MeanoverTime()(reshape_med_1)
    lstm_med_temp = LSTM(mid_dim, )(max_med)

    dense_softmax = Dense(nb_classes, name='dense_softmax', )(lstm_med_temp)
    dense_softmax = Dropout(0.3)(dense_softmax)
    dense_softmax = Activation('softmax')(dense_softmax)
    model = Model(input=input_med, output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()
    return model

def max_model():
    input_med = Input(shape=(maxlen * max_med_len,), name='input_med')
    mask_med = Masking(mask_value=0)(input_med)
    em_med = Embedding(len_medicine_list + 1, output_dim=mid_dim, )(mask_med)
    dropout_med = Dropout(0.5)(em_med)

    reshape_med_1 = Reshape((maxlen, max_med_len, mid_dim))(dropout_med)
    max_med = MaxPooling2D(pool_size=(max_med_len,1))(reshape_med_1)
    reshape_med_2=Reshape((maxlen,mid_dim))(max_med)
    lstm_med_temp = LSTM(mid_dim, )(reshape_med_2)

    dense_softmax = Dense(nb_classes, name='dense_softmax', )(lstm_med_temp)
    dense_softmax =Dropout(0.3)(dense_softmax)
    dense_softmax =Activation('softmax')(dense_softmax)
    model = Model(input=input_med, output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()
    return model


def test_model():
    model =Sequential()
    model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
    model.add(TimeDistributed(Dense(32)))
    model.summary()

def get_nb_lines(nb):
    f_tiwen = open('tiwen_yongyao_infor.dat')
    return_list=[]
    for i, lines in enumerate(f_tiwen):
        line = lines.strip().split(' ')[0:]
        if line[0]!=nb:
            continue
        temp = []
        for item in line:
            temp.append(item)
        return_list.append(temp)
    return return_list


def nb_get_temp(nb_train):
    med_list=[]
    temp_list=[]
    for i in range(0,len(nb_train)):
    # for i in range(0,20):
        list=[]
        temp_temp_list=[]
        nb=nb_train[i]
        process_list =get_nb_lines(nb)

        min_length=min(len(process_list),maxlen)

        begin_time = datetime.strptime(process_list[0][2]+' '+process_list[0][3], "%Y/%m/%d %H:%M")
        temp_temp_list.append(process_list[0][1])
        temp_med_list = []
        if len(process_list[0])==4:
            temp_med_list.append(int(len_medicine_list))
        else:
            for l in range(4, len(process_list[0])):
                temp_med_list.append(int(process_list[0][l]))
        list.append(temp_med_list)
        flag=True
        count =0
        for k in range(1 ,min_length):
            end_time = datetime.strptime(process_list[k][2]+' '+process_list[k][3], "%Y/%m/%d %H:%M")
            time = (end_time - begin_time).days
            if time < 5:
                temp_med_list = []
                if len(process_list[k]) == 4:
                    temp_med_list.append(int(len_medicine_list))
                else:
                    for l in range(4, len(process_list[k])):
                        temp_med_list.append(int(process_list[k][l]))
                list.append(temp_med_list)
                temp_temp_list.append(process_list[k][1])
            else:
                flag=False
                count=k
                break
        if min_length<maxlen or (min_length==maxlen and flag==False):
            if flag==True:
                for l in range(min_length,maxlen):
                    temp_med_list=[]
                    temp_med_list.append(int(len_medicine_list))
                    list.append(temp_med_list)
                    temp_temp_list.append(process_list[min_length-1][1])
            else:
                for l in range(count,maxlen):
                    temp_med_list=[]
                    temp_med_list.append(int(len_medicine_list))
                    list.append(temp_med_list)
                    temp_temp_list.append(process_list[count-1][1])
        #med list
        med_list.append(list)
        #temp_list
        temp_list.append(temp_temp_list)
    return temp_list,med_list

if __name__=='__main__':
    # test_model()
    # model =preposed_model()
    # model =sim_model()
    # model =max_model()
    # model =mean_model()

    for mn in range(0,3):
        if mn ==0:
            model =preposed_model()
        elif mn==1:
            model =max_model()
        elif mn ==2:
            model =mean_model()
        # split train set and test set
        number, label = read_split('number_category.csv')
        nb_train, nb_test, label_train, label_test = cross_validation.train_test_split(number, label, test_size=0.2)

        # prepare y_train
        label_train = category_to_target(label_train)

        # prepare x_train
        temp_train_list, med_train_list = nb_get_temp(nb_train)
        train_list = []
        temp_train = []
        for k in range(0, len(nb_train)):
            # for k in range(0,10):
            for i in range(0, len(med_train_list[k])):
                temp_train = pad_sequences(med_train_list[k], maxlen=max_med_len, padding='post', truncating='post',
                                           dtype='int64', value=0)
                train_list.append(temp_train[i])
        train_list = np.array(train_list)
        print(train_list.shape)

        input_train_med = []
        for i in range(0, int(len(train_list) / maxlen)):
            temp_med = []
            for k in range(0, maxlen):
                for j in range(0, max_med_len):
                    temp_med.append(train_list[i * maxlen + k][j])
            input_train_med.append(temp_med)
        input_train_med = np.array(input_train_med)
        print(input_train_med.shape)
        print(input_train_med[2])

        # begin training..
        print("Trainging...")
        model.fit(input_train_med, label_train, batch_size=batch_size,
                  nb_epoch=epochs, validation_split=0.05, )

        print("testing...")
        temp_test_list, med_test_list = nb_get_temp(nb_test)
        test_list = []
        temp_test = []
        for k in range(0, len(nb_test)):
            # for k in range(0, 1):
            for i in range(0, len(med_test_list[k])):
                temp_test = pad_sequences(med_test_list[k], maxlen=max_med_len, padding='post', truncating='post',
                                          dtype='int64', value=0)
                test_list.append(temp_test[i])
        test_list = np.array(test_list)
        # print(list)

        input_test_med = []
        for i in range(0, int(len(test_list) / maxlen)):
            temp_med = []
            for k in range(0, maxlen):
                for j in range(0, max_med_len):
                    temp_med.append(test_list[i * maxlen + k][j])
            input_test_med.append(temp_med)
        input_test_med = np.array(input_test_med)
        print(input_test_med.shape)

        # prepare y_test
        label_test = category_to_target(label_test)
        score, acc = model.evaluate(input_test_med, label_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        f=open('result.out','a')
        f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
        f.write('\n')
        if mn==0:
            f.write('preposed model\n')
        elif mn==1:
            f.write('max model\n')
        elif mn==2:
            f.write('mean model\n')
        f.write('Test score: %.4f\n'%( score))
        f.write('Test accuracy: %.4f\n'%( acc))



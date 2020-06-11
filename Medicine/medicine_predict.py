from keras.models import Model
import numpy as np
import csv
import keras.backend as K
from datetime import datetime,timedelta
import random
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers import Masking,Embedding
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import LSTM, Dense,Dropout,MaxPooling1D,MaxPooling2D,Activation,RepeatVector
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.engine import Layer
np.random.seed(1337)
batch_size = 16
epochs = 50
nb_classes =3
maxlen =20
in_file_length =73
max_med_len=10
learning_rate=0.01

len_medicine_list =317    #plus 1 for no medicine used
mid_dim =128



def diff_length(filename):
    csv_reader = csv.reader(open(filename))
    csv1 = []
    for i in csv_reader:
        temp = []
        a = 1
        while a < len(i):
            temp.append(float(i[a]))
            a += 1
        csv1.append(temp)
    return csv1


def fun2(filename):
    csv_reader=csv.reader(open(filename))
    csv1=[]
    for i in csv_reader:
        temp=[]
        a=1
        while a<len(i):
            temp.append(float(i[a]))
            a+=1
        csv1.append(temp)
    return np.array(csv1,dtype=type(csv1[0][0]))

def read_split(filename):
    f_in =open(filename)
    label,number=[],[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number.append(line[0])
        label.append(line[1])
    return np.array(number),np.array(label)

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
        temp_list =nor_list(temp_list)
    return temp_list,med_list

def nor_list(temp_list):
    for k in range(0,len(temp_list)):
        max_item =float(max(temp_list[k]))
        min_item =float(min(temp_list[k]))
        for i in range(0,len(temp_list[k])):
            if max_item==min_item:
                temp_list[k][i]=1.00
            else:
                temp_list[k][i]=(float(temp_list[k][i])-min_item)/(max_item-min_item)
    return temp_list


def read_para(nb_train):
    f_in =open('number_age_col71tran.csv')
    return_list=[]
    for j in range(0, len(nb_train)):
        temp_list = []
        for i, lines in enumerate(f_in):
            line = lines.strip().split(',')[0:]
            if int(nb_train[j])!=int(float(line[0])):
                continue
            temp = []
            for item in line[1:]:
                temp.append(item)
            temp_list.append(temp)
        return_list.append(temp_list)
    print(return_list)
    return np.array(return_list)


def rewrite_read_para(nb_train):
    f_in =open('number_age_col71tran.csv')
    return_list=[]
    for i, lines in enumerate(f_in):
        line=lines.strip().split(',')[0:]
        for j in range(0,len(nb_train)):
            if int(nb_train[j]) != int(float(line[0])):
                continue
            temp = []
            for item in line[1:]:
                temp.append(item)
            return_list.append(temp)
    return np.array(return_list)

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


def prepare_temp(temp_train_list):
    print("main")


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



def medicine_model():
    input_med = Input(shape=(maxlen*max_med_len,), name='input_med')
    mask_med =Masking(mask_value=-1,)(input_med)
    em_med=Embedding(len_medicine_list+1,output_dim=mid_dim)(mask_med)
    dropout_med=Dropout(0.5)(em_med)

    reshape_med_1=Reshape((maxlen,max_med_len,mid_dim))(dropout_med)
    # mean
    # max_med=MeanoverTime()(reshape_med_1)
    # max
    max_med =MaxPooling2D(pool_size=(max_med_len,1))(reshape_med_1)
    max_med=Reshape((maxlen,mid_dim))(max_med)

    input_temp = Input(shape=(maxlen, ), name='input_temp')
    # reshape_temp=Reshape((maxlen,1))(input_temp)
    reshape_temp =RepeatVector(20)(input_temp)

    # merge_med_temp = merge([reshape_temp,max_med], mode='concat', concat_axis=2)
    merge_med_temp = merge([reshape_temp,max_med,], mode='dot',dot_axes=1)
    lstm_med_temp = LSTM(32, )(merge_med_temp)

    input_para = Input(shape=(73,), name='input_papa')
    dense_para = Dense(128, name='dense_para',)(input_para)
    dropout_para=Dropout(0.25)(dense_para)
    dropout_para =Activation('relu')(dropout_para)

    merge_temp_para = merge([dropout_para, lstm_med_temp], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, name='dense_softmax',)(merge_temp_para)
    dense_softmax =Activation('softmax')(dense_softmax)
    model = Model(input=[input_med, input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    model = medicine_model()

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
                                       dtype='int64', value=-1)
            train_list.append(temp_train[i])
    train_list = np.array(train_list)
    print(train_list.shape)
    print(train_list[0])
    print(train_list[1])
    print(train_list[2])

    input_train_med = []
    for i in range(0, int(len(train_list) / maxlen)):
        temp_med = []
        for k in range(0, maxlen):
            for j in range(0, max_med_len):
                temp_med.append(train_list[i * maxlen + k][j])
        input_train_med.append(temp_med)
    input_train_med = np.array(input_train_med)
    print(input_train_med.shape)
    print(input_train_med[0])

    input_train_para = rewrite_read_para(nb_train)
    print(input_train_para.shape)

    # prepare temp
    # temp_train_list=prepare_temp(temp_train_list)




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

    input_test_para = rewrite_read_para(nb_test)
    print(input_test_para.shape)

    for i in range(0,epochs):
        # begin training..
        print("Trainging...")
        model.fit([input_train_med, np.array(temp_train_list), input_train_para], label_train, batch_size=batch_size,
                  nb_epoch=1, validation_split=0.05, )

        score, acc = model.evaluate([input_test_med, np.array(temp_test_list), input_test_para], label_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)




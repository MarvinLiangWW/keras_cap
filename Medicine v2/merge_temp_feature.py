import numpy as np
import csv
import random
from keras.models import Sequential
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers import Masking
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent
from keras.layers import Activation
from keras.optimizers import Adam
from keras.regularizers import l2
# import matplotlib.pyplot as plt
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
RNN = recurrent.LSTM
maxlen =50
in_file_length =87
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
data_version=1

def fun2(filename):
    csv_reader=csv.reader(open(filename))
    csv1=[]
    for i in csv_reader:
        temp=[]
        a=0
        while a<len(i):
            temp.append(float(i[a]))
            a+=1
        csv1.append(temp)
    return np.array(csv1,dtype=type(csv1[0][0]))


def diff_length_csv(filename):
    csv_reader = csv.reader(open(filename))
    csv1 = []
    for i in csv_reader:
        temp = []
        a = 0
        while a < len(i):
            temp.append(float(i[a]))
            a += 1
        csv1.append(temp)
    return csv1

def diff_length_dat(filename):
    f_in = open(filename)
    return_list=[]
    for i,lines in enumerate(f_in):
        line = lines.strip().split(' ')[0:]
        return_list.append(line)
    return return_list


def reshape_dataset(train):
    trainX=np.reshape(train,(train.shape[0],train.shape[1],1))
    return np.array(trainX)


def category_to_target(category):
    y =[]
    for i in range(0,category.shape[0]):
        temp=[]
        for k in range(0,nb_classes):
            if k+1==category[i]:
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
    return np.array(y,dtype=type(y[0][0]))

def split_data(X,X2,y,test_size):
    x_train=[]
    x2_train=[]
    y_train=[]
    x_test=[]
    x2_test=[]
    y_test=[]
    for i in range(0,1277):
        random_i = random.uniform(0,1)
        if random_i<test_size:
            x_test.append(X[i,:])
            x2_test.append(X2[i,:])
            y_test.append(y[i])
        else:
            x_train.append(X[i,:])
            x2_train.append(X2[i, :])
            y_train.append(y[i,:])
    return np.array(x_train),np.array(x2_train),np.array(y_train),np.array(x_test),np.array(x2_test),np.array(y_test)

# x_train, x2_train, y_train, x_test, x2_test, y_test = devide_data(X,X2,y,test_size=0.2)

def devide_data(X,X2,y,test_size):
    x_train = []
    x2_train = []
    y_train = []
    x_test = []
    x2_test = []
    y_test = []
    for x,x2,y_value in zip(X,X2,y):
        random_i = random.uniform(0, 1)
        if random_i < test_size:
            x_test.append(x)
            x2_test.append(x2)
            y_test.append(y_value)
        else:
            x_train.append(x)
            x2_train.append(x2)
            y_train.append(y_value)
    return np.array(x_train), np.array(x2_train), np.array(y_train), np.array(x_test), np.array(x2_test), np.array(
        y_test)


def sequential_model():
    model1 = Sequential()
    model1.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    model1.add(LSTM(16,))
    model1.add(Dropout(0.1))

    model2 = Sequential()
    model2.add(Dense(64, input_dim=in_file_length))
    model2.add(Dropout(0.25))
    model2.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0015),
                  metrics=['accuracy'])
    model.summary()
    return model


def temp_lstm(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)
    print(X.shape)
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat'%(data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_validation = fun3(f_in='nb_x_test_%d.dat'%(data_version))
    x_train, x_test,x_validation, y_train, y_test,y_validation = get_train_test_data(X,y,nb_x_train,nb_x_test,nb_x_validation)
    probability_validation = (sum(y_validation) - len(y_validation)) / len(y_validation)
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_validation:', probability_validation)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    y_validation = category_to_target(y_validation)

    input_temp=Input(shape=(temp_length,),name='input_temp')
    reshapre_temp=Reshape(target_shape=(temp_length,1),name='reshape_temp')(input_temp)
    lstm_temp=LSTM(16,name='lstm_temp')(reshapre_temp)
    dense_lstm=Dense(16,activation='relu')(lstm_temp)
    dropout_lstm=Dropout(0.1)(dense_lstm)
    dense_class =Dense(nb_classes,activation='softmax')(dropout_lstm)

    model =Model(input=input_temp,output=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])
    model.summary()

    best_epoch = 0
    best_acc = probability_validation
    acc_list=[]
    score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,shuffle=True)
        score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
        acc_list.append(acc)
        print('Test score:', score)
        print('Test accuracy:', acc)
        if acc>best_acc:
            best_acc=acc
            model.save_weights('temp_dataset_%d_epoch_%d.h5'%(data_version,epoch))
            if best_epoch!=0:
                os.remove('temp_dataset_%d_epoch_%d.h5'%(data_version,best_epoch))
            best_epoch=epoch
    plt.plot(range(0,nb_epochs+1),acc_list)
    model.load_weights('temp_dataset_%d_epoch_%d.h5' % (data_version, best_epoch))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('probabitity_test:', probability_test)
    print('Test accuracy:', acc)
    # plt.show()
    return model



def model_model():

    input_temp = Input(shape=(temp_length, ), name='input_temp',)
    reshape_temp=Reshape(target_shape=(temp_length, 1),)(input_temp)
    lstm_temp = LSTM(output_dim=16,  name='lstm_temp',trainable=True)(reshape_temp)
    dense_lstm = Dense(16, activation='relu')(lstm_temp)
    dropout_lstm = Dropout(0.1)(dense_lstm)
    # dropout_temp =Dropout(0.1)(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para =Dense(16,activation='relu')(input_para)
    dropout_para = Dropout(0.1)(dense_para)

    # merge_temp_para = merge([input_para, lstm_temp], mode='concat', concat_axis=1)
    merge_temp_para = merge([dropout_lstm, dropout_para], mode='concat', concat_axis=1)
    #dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax',W_regularizer=None)(merge_temp_para)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax',)(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])
    model.summary()
    return model

def fun3(f_in='nb_x_train.dat'):
    f_in=open(f_in)
    nb_x=[]
    for i,lines in enumerate(f_in):
        line =lines.strip()
        nb_x.append(line)
    return nb_x

def get_train_test_data(X_con,y,nb_x_train,nb_x_test,nb_x_validation):
    X_train,X_test,X_validation,y_train,y_test,y_validation=[],[],[],[],[],[]
    for m in range(0,len(nb_x_train)):
        for n in range(0,len(X_con)):
            if float(nb_x_train[m])==float(X_con[n][0]):
                X_train.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_train)):
        for n in range(0,len(y)):
            if float(nb_x_train[m])==float(y[n][0]):
                y_train.append(y[n][1:])
                break
    for m in range(0,len(nb_x_test)):
        for n in range(0,len(X_con)):
            if float(nb_x_test[m])==float(X_con[n][0]):
                X_test.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_test)):
        for n in range(0,len(y)):
            if float(nb_x_test[m])==float(y[n][0]):
                y_test.append(y[n][1:])
                break
    for m in range(0,len(nb_x_validation)):
        for n in range(0,len(X_con)):
            if float(nb_x_validation[m])==float(X_con[n][0]):
                X_validation.append(X_con[n][1:])
                break
    for m in range(0,len(nb_x_validation)):
        for n in range(0,len(y)):
            if float(nb_x_validation[m])==float(y[n][0]):
                y_validation.append(y[n][1:])
                break
    return np.array(X_train),np.array(X_test),np.array(X_validation),np.array(y_train),np.array(y_test),np.array(y_validation)


if __name__ =='__main__':
    # need to change function fun2
    # in detail change parameter a from 1 to 0
    # for m in range(1, 6):
    #     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_con, y, test_size=0.2)
    #     f_out_1 = open('nb_x_train_' + str(m) + '.dat', 'w')
    #     f_out_2 = open('nb_x_test_' + str(m) + '.dat', 'w')
    #     for i in range(0, len(X_train)):
    #         f_out_1.write('%s\n' % (X_train[i][0]))
    #     for i in range(0, len(X_test)):
    #         f_out_2.write('%s\n' % (X_test[i][0]))

    X = diff_length_dat('5_day_15_nor.dat')
    X = np.array(X)
    X2 = fun2('number_age_col85tran.csv')
    y = fun2('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)

    nb_x_train =fun3(f_in='nb_train_%d.dat'%(data_version))
    nb_x_test =fun3(f_in='nb_test_%d.dat'%(data_version))
    nb_x_validation =fun3(f_in='nb_validation_%d.dat'%(data_version))

    X_train,X_test,X_validation,y_train,y_test,y_validation=get_train_test_data(X_con,y,nb_x_train,nb_x_test,nb_x_validation)
    print(X_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length+1:]
    x2_train=X_train[:,0:in_file_length]

    x_test = X_test[:, in_file_length+1:]
    x2_test =X_test[:,0:in_file_length]

    x_validation =X_validation[:,in_file_length+1:]
    x2_validation =X_validation[:,0:in_file_length]

    print((sum(y_test) - len(y_test)) / len(y_test))
    y_train=category_to_target(y_train)
    y_test =category_to_target(y_test)
    y_validation =category_to_target(y_validation)

    model =model_model()
    model_temp =temp_lstm(f_in='5_day_15_nor.dat')
    weights=model_temp.get_layer(name='lstm_temp').get_weights()
    model.get_layer('lstm_temp').set_weights(weights=weights)

    print('Train...')
    best_epoch = 0
    best_acc = 0.0
    acc_list = []
    score, acc = model.evaluate([x_validation,x2_validation], y_validation, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        print('Train...')
        # print(model.get_layer(name='lstm_temp'))
        model.fit([x_train,x2_train], y_train, batch_size=batch_size, nb_epoch=1, )
        score, acc = model.evaluate([x_validation,x2_validation], y_validation, batch_size=batch_size)
        acc_list.append(acc)
        # print('Test score:', score)
        print('Test accuracy:', acc)

        if acc>best_acc:
            best_acc=acc
            model.save_weights('temp_dataset_%d_epoch_%d.h5'%(data_version,epoch))
            if best_epoch!=0:
                os.remove('temp_dataset_%d_epoch_%d.h5'%(data_version,best_epoch))
            best_epoch=epoch


    plt.plot(range(0, nb_epochs + 1), acc_list)
    results = ""
    for i, acc in enumerate(acc_list):
        if acc > 0.730:
            if acc > 0.745:
                results += '\033[1;31m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
            else:
                results += '\033[1;34m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
        else:
            results += str(i + 1) + ':' + str(acc) + '; '
    print(results)
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    plt.show()



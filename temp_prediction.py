import csv
from main import fun2
import os
from main import category_to_target
from main import reshape_dataset
from merge_temp_feature import get_train_test_data
from merge_temp_feature import fun3
from keras.models import Sequential
from keras.layers import Masking,Flatten
from keras.layers import LSTM, Dense,Merge,Dropout,MaxPooling1D,TimeDistributedDense,Convolution1D,SimpleRNN
from keras.layers import recurrent
from keras.optimizers import Adam,Adadelta,Adagrad
from keras.layers import Activation,Reshape
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from scipy.fftpack import fft
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2,l1
import numpy as np
# import matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
# import theano
# theano.config.blas.ldflags='-LC:\\OpenBLAS-v0.2.19-Win64-int32\\bin -lopenblas'
np.random.seed(1337)
batch_size = 16
nb_epochs = 400
nb_classes =2
RNN = recurrent.LSTM
maxlen =50
temp_length =15
nb_filter =10
data_version=2

def temp_length_analysis():
    f_in =open('5_day_tiwencheck.csv')
    y =[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[1:]
        y.append(len(line))
    y.sort()
    k=np.array(np.zeros((1,120)))
    print(k.shape)
    for i in y:
        k[0][i]+=1
    print(k)
    plot(range(0,120),k[0])
    xlabel('5_day_nb_temp')
    ylabel('frequency')
    show()
    # print(y)

# temp_length_analysis()

def normalization(f_in ='5_day_25_check.csv',f_out='5_day_25_nor.dat'):
    f_in= open(f_in)
    f_out =open(f_out,'w')
    for i ,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        number =line[0]
        temp =line[1:]
        max_temp =float(max(temp))
        min_temp =float(min(temp))
        f_out.write(number)
        if max_temp == min_temp:
            for temp_i in temp:
                f_out.write(' 1.000')
        else:
            for temp_i in temp:
                f_out.write(' %.3f'%((float(temp_i)-min_temp)/(max_temp-min_temp)))
        f_out.write('\n')




def temp_prediction():
    X = diff_length_csv('5_day_tiwencheck.csv')
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', dtype='float')
    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    x_train = reshape_dataset(X_train)
    x_test = reshape_dataset(X_test)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(maxlen, 1)))
    model.add(LSTM(16))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes,b_regularizer=l1(0.01)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(model.predict_proba(reshape_dataset(X)))
    print('Test score:', score)
    print('Test accuracy:', acc)

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

def pure_temp_prediction():
    X=diff_length_csv('tiwencheck.csv')
    X=np.array(X)
    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    x_train = reshape_dataset(X_train)
    x_test = reshape_dataset(X_test)
    model = Sequential()
    model.add(LSTM(16,input_shape=(12,1)))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes,bias=False))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.05)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(model.predict_proba(reshape_dataset(X)))
    print('Test score:', score)
    print('Test accuracy:', acc)


def length_temp_prediction():
    X = diff_length_csv('5_day_50_check.csv')
    # X=fun2('5_day_50_check.csv')
    # X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', dtype='float')
    X = np.array(X, dtype=float)

    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    x_train = reshape_dataset(X_train)
    x_test = reshape_dataset(X_test)

    model = Sequential()
    # model.add(Masking(mask_value=0, input_shape=(maxlen, 1)))
    model.add(LSTM(3, input_shape=(maxlen, 1), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Reshape((maxlen * 3,)))
    model.add(Dense(nb_classes, b_regularizer=l1(0.01)))
    model.add(Dropout(0.25))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(model.predict_proba(reshape_dataset(X)))
    print('Test score:', score)
    print('Test accuracy:', acc)


def dense_temp(f_in):
    X = diff_length_dat(f_in)
    X = np.array(X, dtype=float)
    y = fun2('number_category.csv')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = Sequential()
    model.add(Dense(nb_classes,input_shape=(15,)))
    model.add(Dropout(0.1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    acc_list = []
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        # print(model.predict_proba(X))
        print('\nTest score:', score)
        print('Test accuracy:', acc)
    plt.plot(range(0, nb_epochs+1), acc_list)
    plt.show()

def diff_length_dat(filename):
    f_in = open(filename)
    return_list=[]
    for i,lines in enumerate(f_in):
        line = lines.strip().split(' ')[0:]
        return_list.append(line)
    return return_list

def temp(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)
    print(X.shape)

    y = fun2('number_category.csv')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = Sequential()
    model.add(Reshape(input_shape=(temp_length,),target_shape=(temp_length,1)))
    model.add(Convolution1D(border_mode='same',filter_length=2,nb_filter=nb_filter))
    model.add(Reshape(target_shape=(temp_length*nb_filter,)))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # model2=Sequential()
    # model2.add(Dense(input_shape=(50,),output_dim=3))
    # model2.add(Merge([model,model2]))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        # print(model.predict_proba(X))
        print('\nTest score:', score)
        print('Test accuracy:', acc)

def temp_model2():
    model =Sequential()
    model.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    #model.add(Convolution1D(border_mode='same',filter_length=3,nb_filter=3))
    model.add(Convolution1D(border_mode='same',filter_length=2,nb_filter=4, activation='relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(Convolution1D(border_mode='same',filter_length=2,nb_filter=4, activation='relu'))
    model.add(MaxPooling1D(pool_length = 2))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2,W_regularizer=l2(0.01),b_regularizer=l2(0.)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model


def temp_model1():
    model =Sequential()
    model.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    #model.add(Convolution1D(border_mode='same',filter_length=3,nb_filter=3))
    model.add(Convolution1D(border_mode='same',filter_length=3,nb_filter=3, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2,W_regularizer=l2(0.01),b_regularizer=l2(0.)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model

def temp_model():
    model =Sequential()
    model.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    model.add(LSTM(16,))
    # model.add(Dense(16,activation='relu',W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
    model.add(Dense(16,activation='relu',))
    model.add(Dropout(0.1))
    model.add(Dense(2,W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
    # model.add(Dense(2,))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005,clipnorm =1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def fft_model():
    model=Sequential()
    model.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    model.add(LSTM(8,))
    model.add(Dense(2,))
    model.add(Dropout(0.1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005,),
                  metrics=['accuracy'])
    model.summary()
    return model

def fft_function(x_train,x_test,x_validation):
    x_train=fft(x_train)
    x_test=fft(x_test)
    x_validation=fft(x_validation)
    print(x_train)
    print(x_train.shape)
    return x_train,x_test,x_validation

def fft_prediction(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)

    y = fun2('number_category.csv')
    nb_x_train =fun3(f_in='nb_train_%d.dat'%(data_version))
    nb_x_test =fun3(f_in='nb_test_%d.dat'%(data_version))
    nb_x_validation =fun3(f_in='nb_validation_%d.dat'%(data_version))
    x_train, x_test,x_validation ,y_train, y_test,y_validation = get_train_test_data(X,y,nb_x_train,nb_x_test,nb_x_validation)
    x_train,x_test,x_validation=fft_function(x_train,x_test,x_validation)
    probability_validation =(sum(y_validation)-len(y_validation))/len(y_validation)
    probability_test =(sum(y_test)-len(y_test))/len(y_test)
    print('probability_validation:',probability_validation)

    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    y_validation = category_to_target(y_validation)

    #get_model
    model =fft_model()

    #evaluation
    best_epoch =0
    best_acc =probability_validation
    acc_list=[]
    score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
    acc_list.append(acc)
    print('Train...')
    for epoch in range(nb_epochs):
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0,nb_epoch=1,shuffle=True)
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

    plot(range(0, nb_epochs + 1), acc_list)
    plot(range(0, nb_epochs + 1), [probability_validation for i in range(0,nb_epochs+1)])
    savefig('fft_temp_dataset_%d.png'%(data_version))
    show()
    print(best_epoch)
    print(best_acc)
    #test
    model.load_weights('temp_dataset_%d_epoch_%d.h5' % (data_version,best_epoch))
    score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)
    print('Test score:', score)
    print('probabitity_test:',probability_test)
    print('Test accuracy:', acc)




def temp_lstm(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)

    y = fun2('number_category.csv')
    nb_x_train =fun3(f_in='nb_train_%d.dat'%(data_version))
    nb_x_test =fun3(f_in='nb_test_%d.dat'%(data_version))
    nb_x_validation =fun3(f_in='nb_validation_%d.dat'%(data_version))
    x_train, x_test,x_validation ,y_train, y_test,y_validation = get_train_test_data(X,y,nb_x_train,nb_x_test,nb_x_validation)
    probability_validation =(sum(y_validation)-len(y_validation))/len(y_validation)
    probability_test =(sum(y_test)-len(y_test))/len(y_test)
    print('probability_validation:',probability_validation)

    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    y_validation = category_to_target(y_validation)
    # print(y_train)

    #get_model
    model =temp_model2()

    #evaluation
    best_epoch =0
    best_acc =probability_validation
    acc_list=[]
    score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
    acc_list.append(acc)
    print('Train...')
    for epoch in range(nb_epochs):
        model.fit(x_train, y_train, batch_size=batch_size, verbose=0,nb_epoch=1,shuffle=True)
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

    plot(range(0, nb_epochs + 1), acc_list)
    print(best_epoch)
    print(best_acc)
    #test
    model.load_weights('temp_dataset_%d_epoch_%d.h5' % (data_version,best_epoch))
    score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)
    print('Test score:', score)
    print('probabitity_test:',probability_test)
    print('Test accuracy:', acc)
    savefig('temp_dataset_%d.png'%(data_version))



def temp_lstm_2(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)
    print(X.shape)
    y = fun2('number_category.csv')
    y=y-1
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    model =Sequential()
    model.add(Reshape(input_shape=(temp_length,), target_shape=(temp_length, 1)))
    model.add(LSTM(16,))
    model.add(Dense(32,))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.002),
                  metrics=['accuracy'])
    model.summary()
    acc_list=[]
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        print(model.predict_classes(X))
        # print(model.predict_proba(X))
        print('\nTest score:', score)
        print('Test accuracy:', acc)
    plt.plot(range(0,nb_epochs+1),acc_list)
    plt.show()


if __name__ =='__main__':
    # temp_lstm(f_in='5_day_15_nor.dat')
    for data_version in range(1,6):
        try:
            fft_prediction(f_in='5_day_15_nor.dat')
        except ValueError:
            continue



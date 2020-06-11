import csv
from main import fun2
from main import category_to_target
from main import reshape_dataset
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import LSTM, Dense,Merge,Dropout,MaxPooling1D,TimeDistributedDense,Convolution1D
from keras.layers import recurrent
from keras.optimizers import Adam
from keras.layers import Activation,Reshape
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2,l1
import numpy as np
# import matplotlib.pyplot as plt

np.random.seed(1337)
batch_size = 16
nb_epochs = 30
nb_classes =3
RNN = recurrent.LSTM
maxlen =50
temp_length =15
nb_filter =10

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
    plt.plot(range(0,120),k[0])
    plt.xlabel('5_day_nb_temp')
    plt.ylabel('frequency')
    plt.show()
    # print(y)



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
        a = 1
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


def dense_temp(f_in='5_day_50_check.csv'):
    X = diff_length_csv(f_in)
    X = np.array(X, dtype=float)
    y = fun2('number_category.csv')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = Sequential()
    model.add(Dense(nb_classes, input_shape=(50,), b_regularizer=l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(model.predict_proba(X))
    print('\nTest score:', score)
    print('Test accuracy:', acc)

def diff_length_dat(filename):
    f_in = open(filename)
    return_list=[]
    for i,lines in enumerate(f_in):
        line = lines.strip().split(' ')[1:]
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
    model.add(Activation('softmax'))

    # model2=Sequential()
    # model2.add(Dense(input_shape=(50,),output_dim=3))
    # model2.add(Merge([model,model2]))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print(model.predict_proba(X))
        print('\nTest score:', score)
        print('Test accuracy:', acc)


if __name__ =='__main__':
    # pure_temp_prediction()
    # temp_prediction()
    # length_temp_prediction()
    # dense_temp('5_day_tiwenlabel.csv')
    # dense_temp('5_day_50_check.csv')
    temp(f_in='5_day_15_nor.dat')
    temp(f_in ='5_day_15_check.csv')
    # normalization()
    # normalization(f_in= '5_day_50_check.csv',f_out='5_day_50_nor.dat')
    # normalization(f_in= '5_day_15_check.csv',f_out='5_day_15_nor.dat')
    # temp_length_analysis()

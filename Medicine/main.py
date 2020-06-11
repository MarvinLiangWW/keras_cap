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
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
np.random.seed(1337)
batch_size = 16
epochs = 40
nb_classes =3
RNN = recurrent.LSTM
maxlen =50
in_file_length =73
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used

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

def diff_length_dat(filename):
    f_in = open(filename)
    return_list=[]
    for i,lines in enumerate(f_in):
        line = lines.strip().split(' ')[1:]
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
    model1.add(Masking(mask_value=0, input_shape=(maxlen, 1)))
    model1.add(LSTM(5,))
    model1.add(Dropout(0.25))

    model2 = Sequential()
    model2.add(Dense(64, input_dim=in_file_length))
    model2.add(Dropout(0.25))
    model2.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model



def model_model():
    input_temp = Input(shape=(maxlen, 1), name='input_temp')
    # mask_temp = Masking(mask_value=-1, name='mask_temp')(input_temp)
    lstm_temp = LSTM(output_dim=5, return_sequences=True, name='lstm_temp')(input_temp)
    reshape_pos = Reshape((maxlen * 5,))(lstm_temp)

    input_para = Input(shape=(73,), name='input_para')
    dense_para = Dense(64, name='dense_para')(input_para)

    merge_temp_para = merge([dense_para, reshape_pos], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax')(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model





if __name__ =='__main__':
    # split_train_test_fun()

    # X=diff_length('5_day_tiwencheck.csv')
    # y=fun2('number_category.csv')
    # X2 = fun2('number_age_col71tran.csv')
    # y = category_to_target(y)

    # X=fun2('tiwencheck.csv')
    # X=reshape_dataset_3_4(X)
    # print(X.shape)
    # print(y.shape)
    # print(X2.shape)
    # y_train = np_utils.to_categorical(y_train, nb_classes)

    # x_train, x2_train, y_train, x_test, x2_test, y_test = split_data(X,X2,y,test_size=0.2)
    # x_train, x2_train, y_train, x_test, x2_test, y_test = devide_data(X,X2,y,test_size=0.2)


    # X = diff_length_csv('5_day_tiwencheck.csv')
    X = diff_length_csv('5_day_15_nor.dat')
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post',value=0,dtype='float')

    X2 = fun2('number_age_col71tran.csv')
    y = fun2('number_category.csv')
    X = np.concatenate((X2, X), axis=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train=category_to_target(y_train)
    y_test =category_to_target(y_test)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length:]
    x2_train=X_train[:,0:len(X2[0])]

    x_test = X_test[:, in_file_length:]
    x2_test =X_test[:,0:len(X2[0])]


    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)
    x_train=reshape_dataset(x_train)
    x_test =reshape_dataset(x_test)
    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)

    model =sequential_model()

    print('Train...')
    model.fit([x_train, x2_train], y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


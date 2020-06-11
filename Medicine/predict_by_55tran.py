import numpy as np
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent
from keras.layers import Activation,RepeatVector,Reshape
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation

from main import diff_length,fun2,category_to_target,reshape_dataset
np.random.seed(1337)
batch_size = 16
epochs = 40
nb_classes =3
maxlen =50
in_file_length =57



def chansfer_file(f_in ='5_day_50_check.csv',into=4,f_out='tiwen_with_category.csv'):
    f_in =open(f_in)
    f_out=open(f_out,'w')
    for i,line in enumerate(f_in):
        line_con=line.strip().split(',')
        for k in range(len(line_con)):
            if k ==0:
                f_out.write('%s'%(line_con[0]))
                continue
            temp=float(line_con[k])
            f_out.write(',%.1f'%(temp))
            if temp<37.2:
                f_out.write(',1,0,0')
            elif 37.2<=temp<38.5:
                f_out.write(',0,1,0')
            elif temp>=38.5:
                f_out.write(',0,0,1')
        f_out.write('\n')



if __name__=='__main__':

    X = diff_length('tiwen_with_category.csv')
    # X = diff_length('5_day_50_check.csv')
    # X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', dtype='float')
    X=np.array(X)

    X2 = fun2('number_age_col55tran.csv')
    y = fun2('number_category.csv')
    X = np.concatenate((X2, X), axis=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    print(y_train.shape)

    # y_train =np.reshape(y_train,(y_train.shape[0],1,y_train.shape[1]))
    # y_test =np.reshape(y_test,(y_test.shape[0],1,y_test.shape[1]))

    # y_train_temp = y_train
    # y_test_temp =y_test
    # for i in range(0,maxlen-1):
    #     y_train=np.concatenate((y_train_temp,y_train),axis=1)
    #     y_test=np.concatenate((y_test_temp,y_test),axis=1)
    # print(y_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length:]
    x2_train = X_train[:, 0:len(X2[0])]

    x_test = X_test[:, in_file_length:]
    x2_test = X_test[:, 0:len(X2[0])]

    print('X_train shape:', x_train.shape)
    print('X_test shape:', x2_train.shape)
    x_train =np.reshape(x_train,(x_train.shape[0],maxlen,4))
    x_test = np.reshape(x_test,(x_test.shape[0],maxlen,4))
    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)
    print(x_train[0])

    model1 = Sequential()
    model1.add(LSTM(output_dim=5,input_shape=(maxlen,4),return_sequences=True))
    model1.add(Dropout(0.3))
    model1.add(Reshape((maxlen*5,)))

    model2 = Sequential()
    model2.add(Dense(64, input_dim=in_file_length))
    model2.add(Dropout(0.3))
    model2.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    model.add(Dense(3,))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit([x_train, x2_train], y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size, verbose=1)
    print('Test score:\n', score)
    print('   Test accuracy:\n', acc)
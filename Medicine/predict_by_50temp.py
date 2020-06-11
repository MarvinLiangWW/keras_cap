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
maxlen =25
in_file_length =57

if __name__=='__main__':
    # X = diff_length('5_day_tiwencheck.csv')
    X = diff_length('5_day_25_check.csv')
    # X=np.array(X)

    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', dtype='float')

    X2 = fun2('number_age_col55tran.csv')
    y = fun2('number_category.csv')
    X = np.concatenate((X2, X), axis=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    print(y_train.shape)

    # y_train =np.reshape(y_train,(y_train.shape[0],1,y_train.shape[1]))
    # y_test =np.reshape(y_test,(y_test.shape[0],1,y_test.shape[1]))
    #
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
    print('X_test shape:', x_test.shape)
    x_train = reshape_dataset(x_train)
    x_test = reshape_dataset(x_test)
    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)

    model1 = Sequential()
    # model1.add(Masking(mask_value=0, input_shape=(maxlen, 1)))
    model1.add(LSTM(output_dim=5,input_shape=(maxlen,1),return_sequences=True))
    model1.add(Dropout(0.25))
    model1.add(Reshape((maxlen*5,)))

    model2 = Sequential()
    model2.add(Dense(64, input_dim=in_file_length))
    model2.add(Dropout(0.25))
    model2.add(Activation('relu'))
    # model2.add(RepeatVector(maxlen))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    hist=model.fit([x_train, x2_train], y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    print(hist.history)
    score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
from main import diff_length
from main import fun2
from main import category_to_target
from main import reshape_dataset
import random
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent
from keras.layers import Activation
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
import numpy as np
np.random.seed(1337)
batch_size = 16
epochs = 40
nb_classes =3
RNN = recurrent.LSTM
maxlen =200
in_file_length =57

if __name__ =='__main__':
    X = fun2('number_age_col55tran.csv')
    y = fun2('number_category.csv')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model =Sequential()
    model.add(Dense(64,input_dim=in_file_length))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=30, validation_split=0.05)
    score, acc = model.evaluate(X_test,  y_test, batch_size=batch_size)
    print(model.predict_proba(X))
    print('Test score:', score)
    print('Test accuracy:', acc)
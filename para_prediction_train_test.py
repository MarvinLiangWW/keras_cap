from accessory import category_to_target,fun2,get_train_test_data,fun3
import random,time
from keras.models import Sequential,Model
from keras.layers import Masking,Input,Reshape
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent
from keras.layers import Activation
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib.pyplot import savefig,plot,legend,show,title,figure,subplot
import os
from temp_prediction_train_test import diff_length_csv,model_model
np.random.seed(1337)
batch_size = 16
nb_epochs = 200
nb_classes =2
RNN = recurrent.LSTM
maxlen =50
in_file_length =87
data_version =2

def get_model(in_file_length):
    model = Sequential()
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    W_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)
                    ))
    # model.add(Dense(nb_classes,W_regularizer=l2(0.01)))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(2))
    # model.add(Dense(nb_classes,input_dim=in_file_length,))
    # model.add(Dropout(0.1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])
    model.summary()
    return model

def para_prediction(f_x='number_age_col85tran_v2.csv',in_file_length=87):
    X = fun2(f_x)
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % data_version)
    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    # get_model
    model = get_model(in_file_length)

    print('Train...')
    acc_list = []
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)

        print('Test score:', score)
        print('Test accuracy:', acc)
    results = ""
    for i, acc in enumerate(acc_list):
        if acc > 0.72:
            if acc > 0.74:
                results += '\033[1;31m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
            else:
                results += '\033[1;34m' + str(i + 1) + ':' + str(acc) + '\033[0m' + '; '
        else:
            results += str(i + 1) + ':' + str(acc) + '; '
    print(results)
    subplot(2,3,data_version)
    title('para_prediction_dataset_%d'%(data_version))
    plot(range(0, nb_epochs + 1), acc_list,label='length:%d'%(in_file_length))
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))

    # show()

if __name__ =='__main__':
    figure()
    for data_version in range(1, 6):
        para_prediction(f_x='number_age_col85tran_v2.csv', )
    show()

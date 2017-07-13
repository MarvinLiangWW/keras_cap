from accessory import category_to_target,fun2,get_train_validation_test_data,fun3
import random
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent
from keras.layers import Activation
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.regularizers import l2
import numpy as np
import os
np.random.seed(1337)
batch_size = 16
nb_epochs = 200
nb_classes =2
RNN = recurrent.LSTM
maxlen =200
in_file_length =87
data_version =2


def get_model():
    model = Sequential()
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    # W_regularizer=l2(0.01),
                    # b_regularizer=l2(0.01)
    ))
    # model.add(Dense(nb_classes,W_regularizer=l2(0.01)))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(2))
    # model.add(Dense(nb_classes,input_dim=in_file_length,W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ =='__main__':
    X = fun2('number_age_col85tran.csv')
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_train_%d.dat' % data_version)
    nb_x_test = fun3(f_in='nb_test_%d.dat' % data_version)
    nb_x_validation = fun3(f_in='nb_validation_%d.dat' % (data_version))
    x_train, x_test, x_validation, y_train, y_test, y_validation = get_train_validation_test_data(X, y, nb_x_train, nb_x_test,
                                                                                       nb_x_validation)
    probability_validation = (sum(y_validation) - len(y_validation)) / len(y_validation)
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_validation:', probability_validation)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    y_validation = category_to_target(y_validation)

    #get_model
    model=get_model()

    print('Train...')
    best_epoch = 0
    best_acc = probability_validation
    acc_list = []
    score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
    acc_list.append(acc)
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.05)
        score, acc = model.evaluate(x_validation, y_validation, batch_size=batch_size)
        acc_list.append(acc)

        print('Test score:', score)
        print('Test accuracy:', acc)

        if acc > best_acc:
            best_acc = acc
            model.save_weights('temp_dataset_%d_epoch_%d.h5' % (data_version, epoch))
            if best_epoch != 0:
                os.remove('temp_dataset_%d_epoch_%d.h5' % (data_version, best_epoch))
            best_epoch = epoch
        if best_epoch==0:
            model.save_weights('temp_dataset_%d_epoch_%d.h5'%(data_version,nb_epochs))
            best_epoch=nb_epochs

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
    plt.plot(range(0, nb_epochs + 1), acc_list)
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    # plt.ylim((0.4,0.8))
    print('best_epoch:', best_epoch)
    print('best_acc', best_acc)
    # test
    model.load_weights('temp_dataset_%d_epoch_%d.h5' % (data_version, best_epoch))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('probabitity_test:', probability_test)
    print('Test accuracy:', acc)

    # plt.show()
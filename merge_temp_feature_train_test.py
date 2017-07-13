import numpy as np
import time
import csv
import random
from accessory import fun2,fun3,category_to_target,diff_length_csv,diff_length_dat
from temp_prediction_train_test import conv_lstm_model
from temp_prediction_train_test import model_model,basic_conv
from keras.models import Sequential
from keras.models import Model
from keras.layers import Reshape,Bidirectional
from keras.layers import Input,merge
from keras.layers import Masking,MaxPooling1D,Flatten
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent,Convolution1D
from keras.layers import Activation
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from accessory import get_train_test_data
import matplotlib
# matplotlib.use('Agg')
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
import os
np.random.seed(133)
batch_size = 16
nb_epochs = 200
nb_classes =2
temp_length=15
RNN = recurrent.LSTM
maxlen =50
in_file_length =10
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
is_conv_t_0_f_2 =2
data_version=2
superpara_a=0.1
input_predict_next_length=10


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

def lstm_model():
    input_temp = Input(shape=(maxlen - 1,), name='input_temp')
    reshapre_temp = Reshape(target_shape=(maxlen - 1, 1), name='reshape_temp')(input_temp)
    masking_temp = Masking(mask_value=2)(reshapre_temp)
    lstm_temp = LSTM(16, name='lstm_temp')(masking_temp)
    # lstm_temp = Bidirectional(LSTM(16, name='lstm_temp'))(masking_temp)
    lstm_temp = Dense(16, activation='relu')(lstm_temp)
    dropout_lstm = Dropout(0.25)(lstm_temp)
    dense_class = Dense(nb_classes, activation='softmax', W_regularizer=l2(0.11), b_regularizer=l2(0.01))(dropout_lstm)

    model = Model(input=input_temp, output=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def temp_lstm(f_in='5_day_50_check.csv'):
    if f_in[-1]=='t':
        X = diff_length_dat(f_in)
    else:
        X = diff_length_csv(f_in)
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', value=is_conv_t_0_f_2, dtype=float)
    X = np.array(X, dtype=float)
    print(X.shape)
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat'%(data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat'%(data_version))
    x_train, x_test,y_train, y_test, = get_train_test_data(X,y,nb_x_train,nb_x_test,)

    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_test:',probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model =lstm_model()

    #test
    acc_list=[]
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,shuffle=True)
        train_score,train_acc =model .evaluate(x_train,y_train,batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    # subplot(2,3,data_version)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    f_out = open('top_K_mean.dat', 'a')
    f_out.write('**********\n')
    f_out.write('temp_model_conv_dataset_%d\n' % (data_version) + time.strftime('%Y_%m_%d %H:%M:%S',
                                                                     time.localtime(time.time())) + '\n')
    f_out.write("top-K mean: %.3f\n" % np.mean(np.array(acc_list[:10])))
    f_out.close()

    # show()
    return model


def merge_lstm_model():

    input_temp = Input(shape=(maxlen-1, ), name='input_temp',)
    reshape_temp=Reshape(target_shape=(maxlen-1, 1),)(input_temp)
    masking_temp=Masking(mask_value=2)(reshape_temp)
    # lstm_temp = LSTM(output_dim=10,  name='lstm_temp',)(masking_temp)
    lstm_temp = Bidirectional(LSTM(output_dim=5,  name='lstm_temp',trainable=True))(masking_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)
    lstm_temp = Dense(16, activation='relu')(lstm_temp)
    dropout_lstm =Dropout(0.25)(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para =Dense(16,activation='relu')(input_para)
    dropout_para = Dropout(0.1)(dense_para)

    # merge_temp_para = merge([input_para, lstm_temp], mode='concat', concat_axis=1)
    merge_temp_para = merge([dropout_lstm, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax',W_regularizer=l2(0.01),b_regularizer=l2(0.01))(merge_temp_para)
    # dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax',)(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005,clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def merge_conv_model():
    input_temp = Input(shape=(maxlen - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(maxlen - 1, 1), )(input_temp)
    l_cov1 = Convolution1D(4, 4, activation='relu')(reshape_temp)
    l_pool1 = MaxPooling1D(4, stride=4)(l_cov1)
    l_flat = Flatten()(l_pool1)
    l_lstm = Dropout(0.1)(l_flat)
    # l_lstm = LSTM(5)(l_pool1)
    # l_lstm=Dropout(0.1)(l_lstm)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dense_para = Dropout(0.1)(dense_para)

    merge_temp_para = merge([l_lstm, dense_para], mode='concat', concat_axis=1)
    # merge_temp_para = merge([l_lstm, dropout_para], mode='mul',dot_axes=-1,)

    # dense_para = Dense(8, activation='relu')(merge_temp_para)
    # dropout_para = Dropout(0.1)(dense_para)
    # dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.001),b_regularizer=l2(0.01))(dropout_para)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.),b_regularizer=l2(0.01))(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.00028, ),
                  metrics=['accuracy'])
    model.summary()
    return model


def para_model():
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
    # model.add(Dense(nb_classes,input_dim=in_file_length,W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    return model

def para_prediction():
    # X = fun2('number_age_col85tran_v2.csv')
    # X = fun2('number_age_col71tran_v2.csv')
    X = fun2('cap_feature_2.csv')
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % data_version)
    # nb_x_validation = fun3(f_in='nb_validation_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    # probability_validation = (sum(y_validation) - len(y_validation)) / len(y_validation)
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    # print('probability_validation:', probability_validation)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    # y_validation = category_to_target(y_validation)

    model = Sequential()
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    W_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)
    ))
    # model.add(BatchNormalization())
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

    print('Train...')
    acc_list = []
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,validation_split=0.05 )
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_loss_list.append(train_score)
        test_loss_list.append(score)
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
    # print(results)
    # subplot(2,3,data_version)
    # plot(range(0,nb_epochs),[probability_test for i in range(0,nb_epochs+1)])
    plot(range(0, nb_epochs), acc_list,label='feature')
    plot(range(0, nb_epochs), train_loss_list,label='train_loss')
    plot(range(0, nb_epochs), test_loss_list,label='test_loss')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    f_out = open('top_K_mean.dat', 'a')
    f_out.write('**********\n')
    f_out.write('para_dataset_%d\n' % (data_version) + time.strftime('%Y_%m_%d %H:%M:%S',
                                                                     time.localtime(time.time())) + '\n')
    f_out.write("top-K mean: %.3f\n" % np.mean(np.array(acc_list[:10])))
    f_out.close()

def merge_prediction():
    X = diff_length_csv('2_5_day_nor_s2.csv')
    # X = diff_length_csv('2_5_day_tiwencheck.csv')
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', value=is_conv_t_0_f_2, dtype=float)
    X = np.array(X, dtype=float)
    print(X.shape)
    # X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    X2 = fun2('cap_feature_2.csv')
    print(X2.shape)
    y = fun2('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)

    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    # nb_x_validation =fun3(f_in='nb_validation_%d.dat'%(data_version))

    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    print(X_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length + 1:]
    x2_train = X_train[:, 0:in_file_length]

    x_test = X_test[:, in_file_length + 1:]
    x2_test = X_test[:, 0:in_file_length]

    # x_validation =X_validation[:,in_file_length+1:]
    # x2_validation =X_validation[:,0:in_file_length]

    print((sum(y_test) - len(y_test)) / len(y_test))
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    # y_validation =category_to_target(y_validation)

    model =merge_lstm_model()
    # model.load_weights('merge_dataset_3_2017_06_13_19_47_48.h5',by_name=True)

    # model_temp =temp_lstm(f_in='2_5_day_nor_s2.csv')
    # weights=model_temp.get_layer(name='lstm_temp').get_weights()
    # model.get_layer('lstm_temp').set_weights(weights=weights)

    print('Train...')
    acc_list = []
    train_score_list=[]
    test_score_list=[]
    # score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
    # acc_list.append(acc)
    for epoch in range(nb_epochs):
        print('Train...')
        # print(model.get_layer(name='lstm_temp'))
        model.fit([x_train, x2_train], y_train, batch_size=batch_size, nb_epoch=1,)
        train_score, train_acc = model.evaluate([x_train, x2_train], y_train, batch_size=batch_size)
        score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        # print('Test score:', score)
        print('Test accuracy:', acc)
    # subplot(2,3,data_version)
    plot(range(0, nb_epochs ), acc_list, label='merge_acc')
    plot(range(0, nb_epochs ), train_score_list, label='train_loss')
    plot(range(0, nb_epochs ), test_score_list, label='test_loss')
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
    legend()
    title('merge_conv_lstm_dataset_%d' % (data_version))
    f_out = open('top_K_mean.dat', 'a')
    f_out.write('**********\n')
    f_out.write('merge_model_lstm_dataset_%d\n' % (data_version) + time.strftime('%Y_%m_%d %H:%M:%S',
                                                                    time.localtime(time.time())) + '\n')
    f_out.write("top-K mean: %.3f\n" % np.mean(np.array(acc_list[:10])))
    f_out.close()
    # savefig('merge_dataset_%d_'%(data_version)+time.strftime('%Y_%m_%d_%H_%M_%S.png',time.localtime(time.time())))
    # model.save('merge_dataset_%d_'%(data_version)+time.strftime('%Y_%m_%d_%H_%M_%S.h5',time.localtime(time.time())))


def A_merge():
    X = diff_length_csv('2_5_day_nor_s2.csv')
    # X = diff_length_csv('2_5_day_tiwencheck.csv')
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', value=is_conv_t_0_f_2, dtype=float)
    X = np.array(X, dtype=float)
    print(X.shape)
    X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    print(X2.shape)
    y = fun2('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)

    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    print(X_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length + 1:]
    x2_train = X_train[:, 0:in_file_length]

    x_test = X_test[:, in_file_length + 1:]
    x2_test = X_test[:, 0:in_file_length]

    print((sum(y_test) - len(y_test)) / len(y_test))
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)



    model_temp =lstm_model()
    model_para =para_model()

    # test
    acc_list = []
    for epoch in range(nb_epochs):
        model_temp.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, shuffle=True)
        temp_predicted =model_temp.predict(x_test,batch_size=batch_size,)
        model_para.fit(x2_train, y_train, batch_size=batch_size, nb_epoch=1, shuffle=True)
        para_predicted =model_para.predict(x2_test,batch_size=batch_size)
        merge_predicted =superpara_a*temp_predicted+(1-superpara_a)*para_predicted
        count =0
        for i in range(0,len(merge_predicted)):
            if round(merge_predicted[i][0])==y_test[i][0]:
                count+=1
        acc_list.append(float(count)/len(y_test))
        print('acc=',float(count)/len(y_test))

    subplot(2, 3, data_version)
    plot(range(0, nb_epochs), acc_list, label='A_merge')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    f_out = open('top_K_mean.dat', 'a')
    f_out.write('**********\n')
    f_out.write('merge_A_dataset_%d\n' % (data_version) + time.strftime('%Y_%m_%d %H:%M:%S',
                                                                                time.localtime(time.time())) + '\n')
    f_out.write("top-K mean: %.3f\n" % np.mean(np.array(acc_list[:10])))
    f_out.close()


def joint_learning_model():
    #temp for over 37.2
    input_temp = Input(shape=(maxlen - 1,), name='input_temp')
    reshapre_temp = Reshape(target_shape=(maxlen - 1, 1), name='reshape_temp')(input_temp)
    masking_temp = Masking(mask_value=2)(reshapre_temp)
    lstm_temp = LSTM(16, name='lstm_temp')(masking_temp)
    dropout_lstm = Dropout(0.1)(lstm_temp)

    #temp for predict the next
    input_temp_next =Input(shape=(input_predict_next_length,),name='input_temp_next')
    reshape_temp_next =Reshape(target_shape=(input_predict_next_length,1),name='reshape_temp_next')(input_temp_next)
    masking_temp_next =Masking(mask_value=2)(reshape_temp_next)
    lstm_temp_next =LSTM(1,return_sequences=True,name ='lstm_temp_next')(masking_temp_next)
    dropout_lstm_next =Dropout(0.1)(lstm_temp_next)
    dense_temp = Dense(1, activation='relu',name='dense_temp')(dropout_lstm_next)

    reshape_dense_temp =Reshape(target_shape=(input_predict_next_length,),name='reshape_dense_temp')(dense_temp)


    #for feature
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16,name='dense_para')(input_para)
    dense_para = Dropout(0.1)(dense_para)
    dense_para= Activation('relu')(dense_para)

    merge_temp_para = merge([reshape_dense_temp,dropout_lstm, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.01),
                          b_regularizer=l2(0.01))(merge_temp_para)


    model = Model(input=[input_temp_next,input_temp,input_para], output=[dense_temp,dense_softmax])
    model.compile(loss={'dense_temp':'mse','dense_softmax':'categorical_crossentropy'},
                  optimizer=Adam(lr=0.001),
                  metrics={'dense_temp':'mse','dense_softmax':'accuracy'})
    model.summary()
    return model

def joint_learning():
    model =joint_learning_model()
    exit()
    X = diff_length_csv('2_5_day_nor_s2.csv')
    # X = diff_length_csv('2_5_day_tiwencheck.csv')
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post', value=is_conv_t_0_f_2, dtype=float)
    X = np.array(X, dtype=float)
    print(X.shape)
    X2 = fun2('number_age_col85tran_v2.csv')
    # X2 = fun2('number_age_col71tran_v2.csv')
    print(X2.shape)
    y = fun2('number_category.csv')
    X_con = np.concatenate((X2, X), axis=1)

    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))
    X_train, X_test, y_train, y_test, = get_train_test_data(X_con, y, nb_x_train, nb_x_test, )
    print(X_train.shape)

    # x_train represents list temperature
    # x2_train represents test parameter
    x_train = X_train[:, in_file_length + 1:]
    x2_train = X_train[:, 0:in_file_length]

    x_test = X_test[:, in_file_length + 1:]
    x2_test = X_test[:, 0:in_file_length]

    print((sum(y_test) - len(y_test)) / len(y_test))
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)



if __name__ =='__main__':

    # figure()
    for data_version in range(3,4):
        # temp_lstm(f_in='2_5_day_nor_s2.csv')
        para_prediction()
        # merge_prediction()
    legend()
    show()



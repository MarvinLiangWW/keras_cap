import numpy as np
import time
from keras.layers import Layer
from keras import initializers
from accessory import same_length_csv,read_case_nb,category_to_target,diff_length_csv,diff_length_dat
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf
# from MeanOverTime import *
from keras.layers import Reshape,Bidirectional
from keras.layers import Input,merge,RepeatVector
from keras.layers import Masking,MaxPooling1D,Flatten,Permute,Lambda
from keras.layers import LSTM, Dense,Merge,Dropout
from keras.layers import recurrent,Convolution1D
from keras.layers import Activation
from keras.optimizers import Adam,SGD,rmsprop
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras import backend as K
from accessory import get_train_test_data
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
# matplotlib.use('Agg')
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure,ylim
from keras.preprocessing.sequence import pad_sequences
SEED=1337
np.random.seed(SEED)
batch_size = 16
nb_epochs = 200
nb_classes =2
time_steps =51
in_file_length =10
data_version=5
padding_value=0

#tranfrom lstm input from x to sqrt(x),x,x*x


def step_change(X):
    tran_x =[]
    for i in range(0, len(X)):
        temp_i=[]
        for j in range(0,len(X[i])):
            temp_x = []
            X[i][j]=float(X[i][j])
            temp_x.append(np.sqrt(abs(X[i][j])))
            temp_x.append(X[i][j])
            temp_x.append(X[i][j]*X[i][j])
            temp_i.append(temp_x)
        tran_x.append(temp_i)
    return np.array(tran_x)

def model_4():
    #tranfrom lstm input from x to sqrt(x),x,x*x
    input_temp = Input(shape=(time_steps - 1,3), name='input_temp',)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(input_temp)
    # lstm_temp =Dropout(0.1,seed=SEED)(lstm_temp)
    lstm_temp = Dense(16, activation='relu')(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED),activation='softmax', name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003,clipnorm=1. ),
                  metrics=['accuracy'])
    model.summary()
    return model

def merge_model_4():
    input_temp = Input(shape=(time_steps - 1,3), name='input_temp', )
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        input_temp)
    lstm_temp = Dense(16, activation='relu')(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.1, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def temp_model_4_study(model):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    print(X.shape)
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    x_train =step_change(x_train)
    x_test =step_change(x_test)

    model =model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
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
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    title('temp_study_%d' % (data_version))

def merge_model_4_study(model):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    print(X.shape)
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    X2 = same_length_csv('cap_feature.csv')
    x2_train, x2_test, y2_train, y2_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )

    x_train = step_change(x_train)
    x_test = step_change(x_test)

    model = model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit([x_train,x2_train], y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate([x_train,x2_train], y_train, batch_size=batch_size)
        score, acc = model.evaluate([x_test,x2_test], y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    title('temp_study_%d' % (data_version))



def model_lstm():
    input_temp = Input(shape=(10,5), name='input_temp', )

    lstm_temp = LSTM(units=5,kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(input_temp)
    lstm_temp =Dense(16)(lstm_temp)
    lstm_temp=Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def model_lstm_2():
    input_temp = Input(shape=(50,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(10, 5), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=5,kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(reshape_temp)
    lstm_temp =Dense(16)(lstm_temp)
    lstm_temp=Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def step_change_5(X,width =5):
    tran_x=[]
    for i in range(0,len(X)):
        temp_i=[]
        for j in range(0,int(len(X[i])/width)):
            temp_i.append(X[i][(0+j*width):(width+j*width)])
        tran_x.append(temp_i)
    return np.array(tran_x)

def temp_mutilstm_study(model):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    print(X.shape)
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    x_train =step_change_5(x_train)
    x_test =step_change_5(x_test)

    model =model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
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
    acc_list_100_110 =acc_list[99:109]
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n'%(data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))

def get_train_fun():
    x=open('front_1_10_temp.csv')
    y=open('front_2_11_temp.csv')
    x_train=[]
    y_train=[]
    for i,lines in enumerate(x):
        line=lines.strip().split(',')[0:]
        x_train.append(line)
    for i,lines in enumerate(y):
        line=lines.strip().split(',')[0:]
        y_train.append(line)
    return np.array(x_train),np.array(y_train)

def reshape_dataset(train):
    trainX=np.reshape(train,(train.shape[0],train.shape[1],1))
    return np.array(trainX)


def gbdt_lstm():
    temp_x, temp_y = get_train_fun()

    X = same_length_csv('cap_feature.csv')
    print(X.shape)
    y = same_length_csv('number_category.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))

    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )

    # y_train = category_to_target(y_train)
    # y_test = category_to_target(y_test)

    temp_1_10_train, temp_1_10_test, temp_2_11_train, temp_2_11_test = get_train_test_data(temp_x, temp_y,nb_x_train, nb_x_test)
    temp_1_10_train = reshape_dataset(np.array(temp_1_10_train))
    temp_2_11_train = reshape_dataset(temp_2_11_train)
    temp_1_10_test = reshape_dataset(temp_1_10_test)
    temp_2_11_test = reshape_dataset(temp_2_11_test)


    input_temp =Input(shape=(10,1),name='input_temp')
    lstm_temp =LSTM(16,return_sequences=True,name='lstm_temp')(input_temp)
    lstm_temp =Dropout(0.25)(lstm_temp)
    dense_temp =Dense(1,name='dense_temp',activation='relu')(lstm_temp)
    model =Model(inputs=input_temp,outputs=dense_temp)
    model.compile(loss={'dense_temp': 'mse'},
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics={'dense_temp': 'mse'},
                  )
    model.summary()

    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer('dense_temp').output)

    score_list=[]
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit([temp_1_10_train], [temp_2_11_train], batch_size=batch_size, epochs=1,shuffle=True,verbose=True)
        train_temp=intermediate_layer_model.predict(temp_1_10_train)
        test_temp=intermediate_layer_model.predict(temp_1_10_test)
        train_temp =np.reshape(train_temp,newshape=(train_temp.shape[0],train_temp.shape[1]*train_temp.shape[2]))
        test_temp =np.reshape(test_temp,newshape=(test_temp.shape[0],test_temp.shape[1]*test_temp.shape[2]))
        x_epoch_train=np.concatenate([train_temp,X_train],axis=1)
        x_epoch_test=np.concatenate([test_temp,X_test],axis=1)
        print(x_epoch_train.shape)
        lr = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
        lr = lr.fit(x_epoch_train, y_train, )
        score = lr.score(x_epoch_test, y_test)
        print(score)
        score_list.append(score)
    plot(range(0, nb_epochs), score_list, label='temp_acc')
    acc_list_100_110 =score_list[99:109]
    acc_list = sorted(score_list, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n'%(data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))


if __name__=='__main__':
    for data_version in range(5,6):
        #feature prediction baseline
        # feature_prediction(data_version)
        # gbdt_lstm()

        #get model
        model=model_lstm()
        # model =merge_att_3_cnn_155_bilstm_5()

        #get temp prediction result
        # temp_mutilstm_study(model)

        #get merge prediction result
        # merge_model_study(model)

        legend()
        ylim((0.5,1.0))
        show()


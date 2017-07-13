import numpy as np
from accessory import fun2,fun3,category_to_target,diff_length_csv,diff_length_dat,reshape_dataset
from keras.models import Model
from keras.layers import Input,merge
from keras.layers import Masking,MaxPooling1D,Flatten,TimeDistributedDense,BatchNormalization
from keras.layers import LSTM, Dense,Merge,Dropout,SimpleRNN,Bidirectional
from keras.layers import Activation
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from accessory import get_train_test_data
np.random.seed(137)
batch_size = 16
nb_epochs = 200
nb_classes =2
temp_length=15
maxlen =50
in_file_length =10
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
is_conv_t_0_f_2 =0
data_version=1
input_temp_length = 10

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


def merge_predict_next_temp_model():
    input_temp =Input(shape=(input_temp_length,1),name='input_temp')
    lstm_temp =LSTM(16,return_sequences=True,name='lstm_temp')(input_temp)
    # lstm_temp =Bidirectional(LSTM(8,return_sequences=True,name='lstm_temp'))(input_temp)
    lstm_temp =Dropout(0.25)(lstm_temp)
    dense_temp =Dense(1,name='dense_temp',activation='relu')(lstm_temp)
    # dense_temp =BatchNormalization()(dense_temp)
    # dense_temp =Dropout(0.2)(dense_temp)
    # dense_ = Dense(5, activation='relu')(lstm_temp)
    flatten_temp=Flatten()(dense_temp)

    # for feature
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, name='dense_para')(input_para)
    dense_para = Dropout(0.1)(dense_para)
    dense_para = Activation('relu')(dense_para)
    # dense_para =BatchNormalization()(dense_para)

    merge_temp_para = merge([flatten_temp, dense_para], mode='concat', concat_axis=1)
    # merge_temp_para =BatchNormalization()(merge_temp_para)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.1),
                          b_regularizer=l2(0.))(merge_temp_para)

    model = Model(input=[input_temp, input_para], output=[dense_temp, dense_softmax])

    model.compile(loss={'dense_temp': 'mse', 'dense_softmax': 'categorical_crossentropy'},
                  optimizer=Adam(lr=0.001, clipnorm=1.),
                  metrics={'dense_temp': 'mse', 'dense_softmax': 'accuracy'},
                  loss_weights={'dense_softmax':0.1, 'dense_temp':1 })
    model.summary()


    return model

if __name__=='__main__':
    model =merge_predict_next_temp_model()

    temp_x, temp_y = get_train_fun()
    # X = fun2('number_age_col85tran_v2.csv')
    # X = fun2('number_age_col71tran_v2.csv')
    X = fun2('cap_feature_2.csv')
    print(X.shape)
    y = fun2('number_category.csv')
    nb_x_train = fun3(f_in='nb_x_train_%d.dat' % (data_version))
    nb_x_test = fun3(f_in='nb_x_test_%d.dat' % (data_version))

    X_train, X_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    print(X_train.shape)

    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    temp_1_10_train, temp_1_10_test, temp_2_11_train, temp_2_11_test = get_train_test_data(temp_x, temp_y,nb_x_train, nb_x_test)
    temp_1_10_train = reshape_dataset(np.array(temp_1_10_train))
    temp_2_11_train = reshape_dataset(temp_2_11_train)
    temp_1_10_test = reshape_dataset(temp_1_10_test)
    temp_2_11_test = reshape_dataset(temp_2_11_test)
    print(temp_1_10_train.shape)

    mse_list=[]
    acc_list=[]
    train_mse_list=[]
    train_acc_list=[]
    train_merge_loss_list=[]
    test_merge_loss_list=[]
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit([temp_1_10_train,X_train], [temp_2_11_train,y_train], batch_size=batch_size, nb_epoch=1,shuffle=True,verbose=True)
        train_whole_loss,train_temp_loss,train_merge_loss,train_mse,train_acc=model.evaluate([temp_1_10_train,X_train], [temp_2_11_train,y_train], batch_size=batch_size)
        whole_loss,temp_loss,merge_loss,mse,acc=model.evaluate([temp_1_10_test,X_test], [temp_2_11_test,y_test], batch_size=batch_size)
        mse_list.append(mse)
        acc_list.append(acc)
        train_merge_loss_list.append(train_merge_loss)
        test_merge_loss_list.append(merge_loss)

    plot(range(0,nb_epochs),acc_list,label='acc')
    plot(range(0,nb_epochs),mse_list,label='mse')
    plot(range(0,nb_epochs),train_merge_loss_list,label='train_merge_loss')
    plot(range(0,nb_epochs),test_merge_loss_list,label='test_merge_loss')

    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print('dataset%d'%(data_version))
    print("top-K mean: %.3f" % np.mean(np.array(acc_list[:10])))
    legend()
    show()
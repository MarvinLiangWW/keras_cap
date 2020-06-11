import numpy as np
from accessory import category_to_target,diff_length_csv,diff_length_dat
from keras.models import Model
from keras.layers import Input,merge
from keras.layers import Masking,MaxPooling1D,Flatten,BatchNormalization
from keras.layers import LSTM, Dense,Merge,Dropout,SimpleRNN,Bidirectional
from keras.layers import Activation
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure,ylim
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from accessory import get_train_test_data
from keras.initializers import glorot_uniform
from accessory import same_length_csv,read_case_nb
SEED =137
np.random.seed(SEED)
batch_size = 16
nb_epochs = 200
nb_classes =2
temp_length=15
maxlen =50
in_file_length =10
max_med_len=10
len_medicine_list =317    #plus 1 for no medicine used
is_conv_t_0_f_2 =0
data_version=5
input_temp_length = 10

def reshape_dataset(train):
    trainX=np.reshape(train,(train.shape[0],train.shape[1],1))
    return np.array(trainX)


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
    flatten_temp=Flatten()(dense_temp)

    # for feature
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, name='dense_para')(input_para)
    dense_para = Dropout(0.1)(dense_para)
    dense_para = Activation('relu')(dense_para)

    merge_temp_para = merge([flatten_temp, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.))(merge_temp_para)

    model = Model(inputs=[input_temp, input_para], outputs=[dense_temp, dense_softmax])

    model.compile(loss={'dense_temp': 'mse', 'dense_softmax': 'categorical_crossentropy'},
                  optimizer=Adam(lr=0.001, clipnorm=1.),
                  metrics={'dense_temp': 'mse', 'dense_softmax': 'accuracy'},
                  loss_weights={'dense_softmax':0.1, 'dense_temp':1 })
    model.summary()

    return model


def new_merge_predict_next_temp_model():
    input_temp =Input(shape=(input_temp_length,1),name='input_temp')
    # lstm_temp =LSTM(10,kernel_initializer=glorot_uniform(seed=SEED),return_sequences=True,name='lstm_temp')(input_temp)
    lstm_temp =Bidirectional(LSTM(5,kernel_initializer=glorot_uniform(seed=SEED),return_sequences=True,name='lstm_temp'))(input_temp)
    lstm_temp =Dropout(0.25)(lstm_temp)
    dense_temp =Dense(1,kernel_initializer=glorot_uniform(seed=SEED),name='dense_temp',activation='relu')(lstm_temp)
    flatten_temp=Flatten()(dense_temp)

    # for feature
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, kernel_initializer=glorot_uniform(seed=SEED),name='dense_para')(input_para)
    dense_para = Dropout(0.1)(dense_para)
    dense_para = Activation('relu')(dense_para)

    merge_temp_para = merge([flatten_temp, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED),activation='softmax', name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.))(merge_temp_para)

    model = Model(inputs=[input_temp, input_para], outputs=[dense_temp, dense_softmax])

    model.compile(loss={'dense_temp': 'mse', 'dense_softmax': 'categorical_crossentropy'},
                  optimizer=Adam(lr=0.001, clipnorm=1.),
                  metrics={'dense_temp': 'mse', 'dense_softmax': 'accuracy'},
                  loss_weights={'dense_softmax':0.1, 'dense_temp':1 })
    model.summary()


    return model



if __name__=='__main__':
    # model =merge_predict_next_temp_model()
    model =new_merge_predict_next_temp_model()

    temp_x, temp_y = get_train_fun()

    X = same_length_csv('cap_feature.csv')
    print(X.shape)
    y = same_length_csv('number_category.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train =read_case_nb(f_in ='nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_case_nb(f_in ='nb_test_cv%d.dat'%(data_version))

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
        model.fit([temp_1_10_train,X_train], [temp_2_11_train,y_train], batch_size=batch_size, epochs=1,shuffle=True,verbose=True)
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

    acc_list_100_110 = acc_list[99:109]
    acc_list_200 = acc_list[0:200]
    print(len(acc_list_200))
    acc_list_210 = acc_list[200:]
    print(acc_list_210)
    print(len(acc_list_210))
    acc_list_sored = sorted(acc_list_200, reverse=True)
    print(acc_list)
    title('merge_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sored[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sored[:50])))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))
    legend()
    ylim((0.5,1.0))
    show()
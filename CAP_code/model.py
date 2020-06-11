import getopt
import sys

from keras.callbacks import EarlyStopping
from keras.layers import Activation, SimpleRNN
from keras.layers import Convolution1D
from keras.layers import Input, merge
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.layers import Masking, MaxPooling1D, Flatten
from keras.layers import Reshape, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from matplotlib.pyplot import plot, legend, show, title, ylim
from sklearn import metrics

from Attention import *
from accessory import get_train_test_data
from accessory import same_length_csv, read_case_nb, category_to_target, diff_length_csv

SEED = 1337
np.random.seed(SEED)
batch_size = 16
nb_epochs = 200
nb_classes = 2
in_file_length = 10


# data_version=5


# lstm unit can be 2,4,5
def model_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)

    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(reshape_temp)
    lstm_temp = Dense(16, )(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_lstm_():
    input_temp = Input(shape=(50,), name='input_temp', )
    # reshape_temp = Reshape(target_shape=(10, 5), name='reshape_temp')(input_temp)
    # reshape_temp = Reshape(target_shape=(25, 2), name='reshape_temp')(input_temp)
    reshape_temp = Reshape(target_shape=(5, 10), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(reshape_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(reshape_temp)
    # lstm_temp = LSTM(units=4,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)

    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.1, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    merge_temp_para = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(merge_temp_para)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_stacklstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=4, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        reshape_temp)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp_2')(lstm_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_simpleRnn():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = SimpleRNN(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(reshape_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.1)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_lstm_flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=2, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        reshape_temp)
    lstm_temp = Flatten()(lstm_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.5)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_lstm_flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    # reshape_temp =Masking(mask_value=-1)(reshape_temp)
    # lstm_temp = LSTM(units=2,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        reshape_temp)
    lstm_temp = Flatten()(lstm_temp)

    lstm_temp = Dense(16)(lstm_temp)
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


def merge_lstm_meanovertime():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    # reshape_temp =Masking(mask_value=-1)(reshape_temp)
    # lstm_temp = LSTM(units=2,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        reshape_temp)
    lstm_temp = MeanoverTime()(lstm_temp)

    lstm_temp = Dense(16)(lstm_temp)
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


def model_bilstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)

    lstm_temp = Bidirectional(LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp'),
                              merge_mode='sum')(reshape_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    lstm_temp = Dense(16, kernel_initializer=glorot_uniform(seed=SEED), )(lstm_temp)
    lstm_temp = Dropout(0.5)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_bilstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)
    lstm_temp = Bidirectional(LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', ),
                              merge_mode='sum')(reshape_temp)
    # lstm_temp = LSTM(units=4,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)

    lstm_temp = Dense(16)(lstm_temp)
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
                  metrics=['accuracy', ])
    model.summary()
    return model


def model_bilstm_flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = Bidirectional(
        LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp'),
        merge_mode='sum')(reshape_temp)
    lstm_temp = Flatten()(lstm_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_bilstm_flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = Bidirectional(
        LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', ),
        merge_mode='sum')(reshape_temp)
    lstm_temp = Flatten()(lstm_temp)
    lstm_temp = Dense(16)(lstm_temp)
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


def model_cnn():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp')
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    l_cov1 = Convolution1D(4, 4, activation='relu')(reshape_temp)
    l_pool1 = MaxPooling1D(4, stride=4)(l_cov1)
    l_flat = Flatten()(l_pool1)
    l_flat = Dropout(0.25)(l_flat)
    dense_class = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.))(l_flat)
    model = Model(inputs=input_temp, outputs=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  # optimizer=SGD(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), )(input_temp)
    l_cov1 = Convolution1D(4, 4, activation='relu')(reshape_temp)
    l_pool1 = MaxPooling1D(4, stride=4)(l_cov1)
    l_flat = Flatten()(l_pool1)
    l_lstm = Dropout(0.1)(l_flat)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dense_para = Dropout(0.1)(dense_para)

    merge_temp_para = merge([l_lstm, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.01),
                          b_regularizer=l2(0.01))(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.00028, ),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp')
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=5, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', name='conv_temp_1')(reshape_temp)
    merge_temp_para = merge([conv_temp_1, reshape_temp], mode='concat', concat_axis=2)
    l_flat = Flatten()(merge_temp_para)
    l_flat = Dropout(0.25)(l_flat)
    dense_class = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.01))(l_flat)
    model = Model(inputs=input_temp, outputs=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_155():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp')
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=5, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', name='conv_temp_1')(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', name='conv_temp_2')(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', name='conv_temp_3')(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    l_flat = Flatten()(merge_temp_para)
    l_flat = Dropout(0.25)(l_flat)
    dense_class = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.01))(l_flat)
    model = Model(inputs=input_temp, outputs=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_111():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp')
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=1, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=1, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    l_flat = Flatten()(merge_temp_para)
    l_flat = Dropout(0.25)(l_flat)
    dense_class = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.01))(l_flat)
    model = Model(inputs=input_temp, outputs=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), )(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    l_flat = Flatten()(merge_temp_para)
    l_flat = Dropout(0.25)(l_flat)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dense_para = Dropout(0.1)(dense_para)

    merge_temp_para = merge([l_flat, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.01),
                          b_regularizer=l2(0.01))(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, ),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_maxp_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp')
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    # conv_temp =Dropout(0.25)(reshape_temp)
    l_conv = Convolution1D(4, 4, activation='relu')(reshape_temp)
    l_pool = MaxPooling1D(4, stride=4)(l_conv)
    l_lstm = LSTM(16)(l_pool)
    dense_class = Dense(nb_classes, activation='softmax', W_regularizer=l2(0.01), b_regularizer=l2(0.01))(l_lstm)
    model = Model(input=input_temp, output=dense_class)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.00025, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_maxp_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), )(input_temp)
    l_cov1 = Convolution1D(4, 4, activation='relu')(reshape_temp)
    l_pool1 = MaxPooling1D(4, stride=4)(l_cov1)
    l_lstm = LSTM(5)(l_pool1)
    l_lstm = Dropout(0.1)(l_lstm)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dense_para = Dropout(0.1)(dense_para)

    merge_temp_para = merge([l_lstm, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, activation='softmax', name='dense_softmax', W_regularizer=l2(0.),
                          b_regularizer=l2(0.01))(merge_temp_para)
    model = Model(input=[input_temp, input_para], output=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.00028, ),
                  metrics=['accuracy'])
    model.summary()
    return model


# lstm output concatenate conv output at axis =1
def model_cnn_concat_lstm_axis_1():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(reshape_temp)
    lstm_temp = Dropout(0.1, seed=SEED)(lstm_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    flatten_temp = Flatten()(conv_temp)
    flatten_temp = Dropout(0.1, seed=SEED)(flatten_temp)

    merge_temp_para = merge([lstm_temp, flatten_temp], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_concat_lstm_axis_1():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(reshape_temp)
    lstm_temp = Dropout(0.1, seed=SEED)(lstm_temp)

    conv_temp = Convolution1D(filters=2, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    max_pooling_temp = MaxPooling1D(pool_size=2, strides=2)(conv_temp)
    flatten_temp = Flatten()(max_pooling_temp)
    flatten_temp = Dropout(0.1, seed=SEED)(flatten_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.1, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, flatten_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


# lstm output concatenate conv output at axis =2
def model_cnn_concat_lstm_axis_2():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        reshape_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([lstm_temp, conv_temp], mode='concat', concat_axis=2)
    merge_temp_para = Flatten()(merge_temp_para)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_concat_lstm_axis_2():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        reshape_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp = merge([lstm_temp, conv_temp], mode='concat', concat_axis=2)
    merge_temp = Flatten()(merge_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.1, seed=SEED)(dense_para)

    merge_temp_para = merge([merge_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_5_lstm():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(merge_temp_para)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_5_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.5, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_5_lstm_Flatten():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)

    lstm_temp = Flatten()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_5_lstm_Flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)
    lstm_temp = Flatten()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

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


def model_cnn_5_lstm_maxovertime():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)

    lstm_temp = MaxoverTime()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_5_lstm_maxovertime():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)
    lstm_temp = MaxoverTime()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_5_lstm_meanovertime():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)

    lstm_temp = MeanoverTime()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_5_lstm_meanovertime():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)
    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)
    lstm_temp = MeanoverTime()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_155_lstm():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(merge_temp_para)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_111_lstm():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=1, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=1, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(merge_temp_para)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_135_lstm_3():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=3, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=3, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(merge_temp_para)
    # lstm_temp =Flatten()(lstm_temp)
    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_135_lstm_3():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=3, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=3, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_149_lstm_Flatten():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=4, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=9, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = Flatten()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_155_lstm_Flatten():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = Flatten()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155_lstm_Flatten():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)
    lstm_temp = Flatten()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.5, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_155_lstm_maxovertime():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = MaxoverTime()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155_lstm_maxovertime():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)

    lstm_temp = MaxoverTime()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_cnn_155_lstm_meanovertime():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = MeanoverTime()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155_lstm_meanovertime():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp', )(
        merge_temp_para)

    lstm_temp = MeanoverTime()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([lstm_temp, dropout_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.1))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def model_att_1_cnn_5_lstm():
    print('Attendtion')
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    att = AttLayer_model_1()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.02),
                          bias_regularizer=l2(0.01))(att)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_1_cnn_5_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    att = AttLayer_model_1()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.5, seed=SEED)(dense_para)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def model_att_1_lstm_5():
    print('Attendtion')
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(reshape_temp)

    att = AttLayer_model_1()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.02),
                          bias_regularizer=l2(0.01))(att)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_1_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(reshape_temp)

    att = AttLayer_model_1()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.5, seed=SEED)(dense_para)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc', 'mse'])
    model.summary()
    return model


def model_att_1_cnn_155_lstm():
    print('Attendtion')
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    att = AttLayer_model_1()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.02),
                          bias_regularizer=l2(0.01))(att)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_1_cnn_155_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    att = AttLayer_model_1()(lstm_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.5, seed=SEED)(dense_para)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_2_lstm_16():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.05)(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(16, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


# can change lstm units from 16 to 5 for a try
def merge_att_2_cnn_5_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.05)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(16, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_2_cnn_155_lstm_16():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.05)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(16, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_2_cnn_155_bilstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = Bidirectional(
        LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
             recurrent_dropout=0.05), merge_mode='sum')(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(5, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_2_cnn_155_lstm_sum():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.05)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(16, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='sum', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_2_lstm_sum():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.05)(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(16, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_2()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='sum', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


# can change lstm units from 5 to 4 for a try
def merge_att_3_cnn_5_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_cnn_1_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp = Convolution1D(filters=1, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                              activation='relu', )(reshape_temp)

    merge_temp_para = merge([reshape_temp, conv_temp], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)

    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01, )(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(time_steps - 1, activation='relu', name='dense_1_para')(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    # att = AttLayer_model_3(name='AttLayer')(in_x)
    att = AttLayer_model_6(name='AttLayer')(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_gru_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)
    lstm_temp = GRU(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                    recurrent_dropout=0.01)(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(time_steps - 1, activation='relu', name='dense_1_para')(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    # att = AttLayer_model_3(name='AttLayer')(in_x)
    att = AttLayer_model_3(name='AttLayer')(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc', ])
    model.summary()
    return model


def merge_att_3_bgru_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = Bidirectional(
        GRU(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
            recurrent_dropout=0.01), merge_mode='sum')(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(time_steps - 1, activation='relu', name='dense_1_para')(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    # att = AttLayer_model_3(name='AttLayer')(in_x)
    att = AttLayer_model_3(name='AttLayer')(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc', 'mse'])
    model.summary()
    return model


def merge_att_3_Blstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=-1)(reshape_temp)

    lstm_temp = Bidirectional(
        LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
             recurrent_dropout=0.01), merge_mode='sum')(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(time_steps - 1, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc', ])
    model.summary()
    return model


def merge_att_3_cnn_155_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=1, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=1, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_cnn_155_bilstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = Bidirectional(
        LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
             recurrent_dropout=0.01), merge_mode='sum')(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_cnn_155_lstm_sum():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='sum', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_3_lstm_sum():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=16, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='sum', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_4_cnn_155_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(merge_temp_para)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.5, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_4()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def merge_att_4_lstm_5():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    lstm_temp = LSTM(units=5, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01)(reshape_temp)

    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_1_para = Dense(50, activation='relu', )(input_para)
    dropout_1_para = Dropout(0.25, seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x = []
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_4()(in_x)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def model_cnn_155_dual_lstm():
    # conv in lstm input
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)

    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=4, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = LSTM(units=4, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp_2')(
        lstm_temp)
    lstm_temp = Flatten()(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model


def merge_cnn_155_dual_lstm():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    conv_temp_1 = Convolution1D(filters=1, kernel_size=1, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_2 = Convolution1D(filters=5, kernel_size=2, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)
    conv_temp_3 = Convolution1D(filters=5, kernel_size=3, kernel_initializer=glorot_uniform(seed=SEED), padding='same',
                                activation='relu', )(reshape_temp)

    merge_temp_para = merge([conv_temp_1, conv_temp_2, conv_temp_3], mode='concat', concat_axis=2)
    lstm_temp = LSTM(units=4, return_sequences=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(
        merge_temp_para)
    lstm_temp = LSTM(units=4, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp_2')(lstm_temp)
    # lstm_temp =Flatten()(lstm_temp)

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


def feature_prediction(data_version):
    X = same_length_csv('cap_feature.csv')
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_cv%d.dat' % (data_version))

    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)

    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = Sequential()
    # model.add(Dense(time_steps-1, input_dim=in_file_length, activation='relu'))
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    W_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)
                    ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    print('Train...')
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_loss_list.append(train_score)
        test_loss_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
        y_pre = model.predict(x_test, )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        print(roc_auc)
        auc_list.append(roc_auc)

    plot(range(0, nb_epochs), acc_list, label='feature')
    plot(range(0, nb_epochs), train_loss_list, label='train_loss')
    plot(range(0, nb_epochs), test_loss_list, label='test_loss')
    acc_list_100_110 = acc_list[99:109]
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 mean: %.3f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(auc_list_sorted[:50])))


def model_study(model, data_version):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    print(X.shape)
    y = same_length_csv('number_category.csv')
    # nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat'%(data_version))
    # nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat'%(data_version))
    nb_x_train = read_case_nb(f_in='nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_cv%d.dat' % (data_version))

    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    print('class 2', (sum(y_test) - len(y_test)) / len(y_test))

    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        y_pre = model.predict(x_test, )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        print(roc_auc)
        auc_list.append(roc_auc)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    plot(range(0, nb_epochs), auc_list, label='auc')
    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    # print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))
    # savefig('temp_study_%d'%(data_version))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 mean: %.3f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(auc_list_sorted[:50])))


def merge_model_study(model, data_version):
    X = diff_length_csv('temperature_37.2_40.0.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_train_cv%d.dat' % (data_version))
    nb_x_test = read_case_nb(f_in='nb_test_cv%d.dat' % (data_version))
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    # use y_test_ for calculate auc

    X2 = same_length_csv('cap_feature.csv')
    x2_train, x2_test, y2_train, y2_test, = get_train_test_data(X2, y, nb_x_train, nb_x_test, )

    model = model

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        if epoch == 100:
            get_3rd_layer_output = K.function(
                [model.get_layer('input_temp').input, model.get_layer('input_para').input, K.learning_phase()],
                [model.get_layer(name='AttLayer').heatmap])
            output = get_3rd_layer_output([x_test, x2_test, 0])
            print(type(output))
            print(output)
            print(output[0].shape)
            f_out = open('heatmap_2.dat', 'w')
            # for i in range(0,len(output[0][0])):
            #     f_out.write('%d' % (i))
            #     for m in range(0,len(output[0])):
            #         f_out.write(' %.5f'%(output[0][m][i]))
            #     f_out.write('\n')
            for i in range(0, len(output[0])):
                f_out.write('%d' % (i))
                for m in range(0, len(output[0][0])):
                    f_out.write(' %.5f' % (output[0][i][m]))
                f_out.write('\n')
            predicted = model.predict([x_test, x2_test], batch_size=batch_size)
            print(predicted.shape)
            f_out = open('att_3_lstm_5_predicted_100.csv', 'w')
            for i in range(0, len(predicted)):
                if predicted[i][0] > 0.5:
                    f_out.write('1\n')
                else:
                    f_out.write('2\n')
            f_out.close()
            exit()

        print('Train...')
        model.fit([x_train, x2_train], y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
        train_score, train_acc = model.evaluate([x_train, x2_train], y_train, batch_size=batch_size, )

        score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
        y_pre = model.predict([x_test, x2_test], )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        print(roc_auc)
        # y_pred =np.zeros(shape=(len(y_pre),))
        # for i in range(0,len(y_pre)):
        #     if y_pre[i][0] >0.5:
        #         y_pred[i]=0
        #     if y_pre[i][1]>0.5:
        #         y_pred[i]=1
        # fpr, tpr, _ = roc_curve((y_test_-1), y_pred)
        # roc_auc = auc(fpr, tpr)
        # print(roc_auc)

        # if epoch ==99:
        #     predict =model.predict([x_test,x2_test],batch_size=batch_size)
        #     predicted =[]
        #     for i in range(0,len(predict)):
        #         if predict[i][0]>=0.5:
        #             predicted.append(1)
        #         else:
        #             predicted.append(2)
        #
        #     f_out =open('merge_att_3_lstm_5_predicted.csv','a')
        #     for i in range(0,len(nb_x_test)):
        #         f_out.write('%s,%d\n'%(nb_x_test[i],predicted[i]))
        #     f_out.close()
        #     exit()
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        auc_list.append(roc_auc)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    plot(range(0, nb_epochs), auc_list, label='auc')
    # acc_list_100_110 =acc_list[99:109]
    acc_list_sorted = sorted(acc_list, reverse=True)
    print(acc_list)
    title('merge_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sorted[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sorted[:50])))
    # print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 mean: %.3f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(auc_list_sorted[:50])))


def model_study_earlystopping(model):
    X = diff_length_csv('temperature.csv')
    X = pad_sequences(X, maxlen=time_steps, padding='post', truncating='post', value=padding_value, dtype=float)
    print(X.shape)
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % data_version)
    x_train, x_test, y_train, y_test, = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)
    model = model

    callbacks = [EarlyStopping(monitor='acc', patience=5)]

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1, callbacks=callbacks)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    acc_list_100_110 = acc_list[99:109]
    acc_list_200 = acc_list[0:200]
    print(len(acc_list_200))
    acc_list_210 = acc_list[200:]
    print(acc_list_210)
    print(len(acc_list_210))
    acc_list_sored = sorted(acc_list_200, reverse=True)
    print(acc_list)
    title('temp_study_%d' % data_version)
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sored[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sored[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))
    # savefig('temp_study_%d'%(data_version))


def merge_model_study_earlystopping(model):
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

    model = model
    callbacks = [EarlyStopping(monitor='acc', patience=2)]

    # test
    acc_list = []
    train_score_list = []
    test_score_list = []
    for epoch in range(nb_epochs):
        print('Train...')
        model.fit([x_train, x2_train], y_train, batch_size=batch_size, epochs=1, shuffle=True, verbose=1,
                  callbacks=callbacks)
        train_score, train_acc = model.evaluate([x_train, x2_train], y_train, batch_size=batch_size)
        score, acc = model.evaluate([x_test, x2_test], y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_score_list.append(train_score)
        test_score_list.append(score)
        print('Test score:', score)
        print('Test accuracy:', acc)
    plot(range(0, nb_epochs), acc_list, label='temp_acc')
    plot(range(0, nb_epochs), train_score_list, label='train_loss')
    plot(range(0, nb_epochs), test_score_list, label='test_loss')
    acc_list_100_110 = acc_list[99:109]
    acc_list_200 = acc_list[0:200]
    print(len(acc_list_200))
    acc_list_210 = acc_list[200:]
    print(acc_list_210)
    print(len(acc_list_210))
    acc_list_sored = sorted(acc_list_200, reverse=True)
    print(acc_list)
    title('temp_study_%d' % (data_version))
    print('temp_study_%d\n' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list_sored[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list_sored[:50])))
    # print("last-10 mean: %.3f" % np.mean(np.array(acc_list_210)))
    print("acc_100-110 mean: %.3f" % np.mean(np.array(acc_list_100_110)))


def para_temp1_model():
    model = Sequential()
    model.add(Dense(16, input_dim=in_file_length, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,
                    W_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)
                    ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    return model


def para_temp_1_prediction():
    # X1 = diff_length_csv('temperature.csv')
    # X1 = pad_sequences(X1, maxlen=20, padding='post', truncating='post', value=padding_value, dtype=float)


    X = same_length_csv('cap_feature.csv')

    # X=np.concatenate((X1,X),axis=1)
    print(X.shape)
    print(X[0])
    y = same_length_csv('number_category.csv')
    nb_x_train = read_case_nb(f_in='nb_x_train_%d.dat' % data_version)
    nb_x_test = read_case_nb(f_in='nb_x_test_%d.dat' % data_version)
    x_train, x_test, y_train, y_test = get_train_test_data(X, y, nb_x_train, nb_x_test, )
    probability_test = (sum(y_test) - len(y_test)) / len(y_test)
    print('probability_test:', probability_test)
    y_train = category_to_target(y_train)
    y_test = category_to_target(y_test)

    model = para_temp1_model()
    print('Train...')
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    auc_list = []
    for epoch in range(nb_epochs):
        # print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1)
        train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        acc_list.append(acc)
        train_loss_list.append(train_score)
        test_loss_list.append(score)
        y_pre = model.predict(x_test, )
        roc_auc = metrics.roc_auc_score(y_test, y_pre)
        auc_list.append(roc_auc)
        print(roc_auc)
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
    plot(range(0, nb_epochs), acc_list, label='feature')
    plot(range(0, nb_epochs), train_loss_list, label='train_loss')
    plot(range(0, nb_epochs), test_loss_list, label='test_loss')

    acc_list = sorted(acc_list, reverse=True)
    print(acc_list)
    print("top-10 mean: %.3f" % np.mean(np.array(acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(acc_list[:50])))
    auc_list_sorted = sorted(auc_list, reverse=True)
    print("top-10 mean: %.3f" % np.mean(np.array(auc_list_sorted[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(auc_list_sorted[:50])))
    print('data_version', data_version)


def merge_mlp():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    input_para = Input(shape=(in_file_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dense_para = Dropout(0.25, seed=SEED)(dense_para)

    merge_temp_para = merge([input_temp, dense_para], mode='concat', concat_axis=1)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.))(merge_temp_para)
    model = Model(inputs=[input_temp, input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def model_mlp():
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    dense_temp = Dense(16, activation='relu')(input_temp)
    dense_temp = Dropout(0.5, seed=SEED)(dense_temp)
    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.11),
                          bias_regularizer=l2(0.05))(dense_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


def usage():
    print('-d data_version')
    print('-m model name')
    print('-h help')


def main_commend():
    opts, args = getopt.getopt(sys.argv[1:], "d:m:h")

    for op, value in opts:
        if op == "-d":
            data_version = value
        elif op == "-h":
            usage()
            sys.exit()
    model = merge_att_3_lstm_5()
    merge_model_study(model, int(data_version))
    print(time_steps)
    # legend()
    # ylim((0.5, 1.0))
    # show()


if __name__ == '__main__':
    time_steps = 51
    padding_value = -1

    # main_commend()
    # exit()
    for data_version in range(5, 6):
        # feature prediction baseline
        feature_prediction(data_version)

        # para_temp_1_prediction()

        # get model
        # model=model_bilstm()
        # model =merge_lstm()
        model = merge_att_3_lstm_5()
        model = model_cnn()
        # model =merge_mlp()

        # get temp prediction result
        model_study(model, data_version)

        # get merge prediction result
        # merge_model_study(model,data_version)


        print(time_steps)

        legend()
        ylim((0.5, 1.0))
        show()

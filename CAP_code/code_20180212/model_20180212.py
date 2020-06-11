from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
from keras.layers import Layer
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf

SEED=1337
np.random.seed(SEED)
nb_classes =2
feature_length =10
PADDING_VALUE=-1
attention_size=5



def model_lstm(time_steps):
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp =Masking(mask_value=PADDING_VALUE)(reshape_temp)

    lstm_temp = LSTM(units=5,kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp')(reshape_temp)
    lstm_temp =Dense(16,kernel_initializer=glorot_uniform(seed=SEED))(lstm_temp)
    lstm_temp=Dropout(0.25)(lstm_temp)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax',)(lstm_temp)
    model = Model(inputs=input_temp, outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003, clipnorm=1.),
                  metrics=['accuracy'])
    model.summary()
    return model

def merge_lstm(time_steps):
    input_temp = Input(shape=(time_steps - 1,), name='input_temp',)
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp = Masking(mask_value=PADDING_VALUE)(reshape_temp)
    lstm_temp = LSTM(units=5,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)
    # lstm_temp = LSTM(units=4,kernel_initializer=glorot_uniform(seed=SEED),name='lstm_temp',)(reshape_temp)

    lstm_temp = Dense(16)(lstm_temp)
    lstm_temp = Dropout(0.25)(lstm_temp)

    input_para = Input(shape=(feature_length,), name='input_para')
    dense_para = Dense(16, activation='relu')(input_para)
    dropout_para = Dropout(0.1, seed=SEED)(dense_para)

    merge_temp_para =merge([lstm_temp,dropout_para],mode='concat',concat_axis=1)
    # merge_temp_para =Dense(16,activation='relu',kernel_regularizer=l2(0.01))(merge_temp_para)
    dense_softmax = Dense(nb_classes,activation='softmax', name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp,input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003,clipnorm=1. ),
                  metrics=['accuracy'])
    model.summary()
    return model


#Attention =x_temp * softmax(uW*tanh(W_temp*x_temp+W_fea*x_fea+ bw))
#model Att_3
class AttLayer_model_3(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
        self.heatmap=[]
        self.supports_masking = True
        super(AttLayer_model_3,self).__init__(**kwargs)

    def build(self, input_shape):
        s_temp =input_shape[0]
        self.W_temp = self.add_weight(shape=(s_temp[-1], self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_temp')
        self.W_fea = self.add_weight(shape=(1,self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_fea')
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True,name='bw')
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = glorot_uniform(seed=SEED), trainable=True,name='uw')
        self.trainable_weights = [self.W_temp,self.W_fea, self.bw, self.uw]
        super(AttLayer_model_3,self).build(input_shape)

    def call(self, x, mask=None):
        x_temp =x[0]  #(none,50,5)
        x_reshaped = tf.reshape(x_temp, [K.shape(x_temp)[0]*K.shape(x_temp)[1], K.shape(x_temp)[-1]]) #(none*50,5)
        temp =K.dot(x_reshaped, self.W_temp) #(none*50,5)

        x_fea =x[1] #(none ,50)
        x_fea =tf.reshape(x_fea,[K.shape(x_fea)[0]*K.shape(x_fea)[1],1]) #(none*50,1)
        fea = K.dot(x_fea, self.W_fea) #(none*50,5)
        intermed =K.tanh(fea+temp+self.bw)
        intermed = tf.reduce_sum(tf.multiply(self.uw, intermed), axis=1)
        # print("intermed",K.shape(intermed))

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x_temp)[0], K.shape(x_temp)[1]]), dim=-1)

        if mask is not None and mask[0] is not None:
            weights =weights *K.cast(mask[0],'float32')
            weights =weights/ K.expand_dims(K.sum(weights,-1),-1)

        weights = tf.expand_dims(weights, axis=-1)
        self.heatmap =weights

        weighted_input = x_temp*weights
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        s =input_shape[0]
        return (s[0],s[2])


#Attention =x_temp*softmax(uW*(tanh(W_temp*x_temp+b_temp).*tanh(W_fea*x_fea+b_fea))+b)
#model_Att_5
class AttLayer_model_5(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
        self.heatmap=[]
        self.supports_masking = True
        super(AttLayer_model_5,self).__init__(**kwargs)

    def build(self, input_shape):
        s_temp =input_shape[0]
        self.W_temp = self.add_weight(shape=(s_temp[-1], self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_temp')
        self.b_temp = self.add_weight(shape=(self.hidden_dim,), initializer='zero', trainable=True, name='b_temp')
        self.W_fea = self.add_weight(shape=(1,self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_fea')
        self.b_fea = self.add_weight(shape=(self.hidden_dim,), initializer='zero', trainable=True, name='b_fea')
        self.b = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True,name='b')
        # print(K.int_shape(self.b))
        self.uw = self.add_weight(shape=(self.hidden_dim,self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='uw')
        self.trainable_weights = [self.W_temp,self.b_temp,self.W_fea, self.b_fea,self.b, self.uw]
        super(AttLayer_model_5,self).build(input_shape)

    def call(self, x, mask=None):
        x_temp =x[0]  #(none,50,5)
        x_reshaped = tf.reshape(x_temp, [K.shape(x_temp)[0]*K.shape(x_temp)[1], K.shape(x_temp)[-1]]) #(none*50,5)
        temp =K.dot(x_reshaped, self.W_temp)+self.b_temp #(none*50,5)

        x_fea =x[1] #(none ,50)
        x_fea =tf.reshape(x_fea,[K.shape(x_fea)[0]*K.shape(x_fea)[1],1]) #(none*50,1)
        fea = K.dot(x_fea, self.W_fea)+self.b_fea #(none*50,5)
        hadamard =K.tanh(temp)*K.tanh(fea)

        intermed = tf.reduce_sum(K.dot(hadamard,self.uw,)+self.b, axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x_temp)[0], K.shape(x_temp)[1]]), dim=-1)


        if mask is not None and mask[0] is not None:
            weights =weights *K.cast(mask[0],'float32')
            weights =weights/ K.expand_dims(K.sum(weights,-1),-1)

        weights = tf.expand_dims(weights, axis=-1)
        self.heatmap =weights

        weighted_input = x_temp*weights
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        s =input_shape[0]
        return (s[0],s[2])



def merge_att_3_lstm_5(time_steps):
    input_temp = Input(shape=(time_steps - 1,), name='input_temp', )
    reshape_temp = Reshape(target_shape=(time_steps - 1, 1), name='reshape_temp')(input_temp)
    reshape_temp =Masking(mask_value=PADDING_VALUE)(reshape_temp)

    lstm_temp = LSTM(units=5, kernel_initializer=True, kernel_initializer=glorot_uniform(seed=SEED), name='lstm_temp',
                     recurrent_dropout=0.01,)(reshape_temp)

    input_para = Input(shape=(feature_length,), name='input_para')
    dense_1_para =Dense(time_steps-1,activation='relu',name='dense_1_para')(input_para)
    dropout_1_para =Dropout(0.5,seed=SEED)(dense_1_para)
    dense_para = Dense(16, activation='relu')(dropout_1_para)
    dropout_para = Dropout(0.25, seed=SEED)(dense_para)

    in_x =[]
    in_x.append(lstm_temp)
    in_x.append(dropout_1_para)
    att = AttLayer_model_3(name='AttLayer')(in_x)
    # att = AttLayer_model_5(name='AttLayer')(in_x)

    # att = Dense(16,)(att)
    # att = Dropout(0.25)(att)

    merge_temp_para = merge([att, dropout_para], mode='concat', concat_axis=1)
    # merge_temp_para = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(merge_temp_para)

    dense_softmax = Dense(nb_classes, kernel_initializer=glorot_uniform(seed=SEED), activation='softmax',
                          name='dense_softmax', kernel_regularizer=l2(0.1),
                          bias_regularizer=l2(0.01))(merge_temp_para)
    model = Model(inputs=[input_temp,input_para], outputs=dense_softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005, clipnorm=1.),
                  metrics=['acc'])
    model.summary()
    return model


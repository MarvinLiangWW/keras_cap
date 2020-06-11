import numpy as np
from keras.layers import Layer
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf
attention_size=5
SEED=1337
np.random.seed(SEED)

#Attention =x_temp * softmax ( uW * tanh( W * x_temp + bw))
#model Att_1
class AttLayer_model_1(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
        super(AttLayer_model_1,self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W')
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True,name='bw')
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = glorot_uniform(seed=SEED), trainable=True,name='uw')
        self.trainable_weights = [self.W, self.bw, self.uw]
        super(AttLayer_model_1,self).build(input_shape)

    def call(self, x, mask=None):
        x_reshaped = tf.reshape(x, [K.shape(x)[0]*K.shape(x)[1], K.shape(x)[-1]])
        ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
        intermed = tf.reduce_sum(tf.multiply(self.uw, ui), axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x)[0], K.shape(x)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x*weights
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

#Attention = x_temp * softmax (uW * tanh((x_temp .* x_fea) + bw))
#model Att_2
class AttLayer_model_2(Layer):

    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
        super(AttLayer_model_2,self).__init__(**kwargs)

    def build(self, input_shape):
        s =input_shape[1]
        self.W = self.add_weight(shape=(1, self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W')
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True,name='bw')
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = glorot_uniform(seed=SEED), trainable=True,name='uw')
        self.trainable_weights = [self.W, self.bw, self.uw]
        super(AttLayer_model_2,self).build(input_shape)

    def call(self, x, mask=None):
        x_temp =x[0]
        x_fea =x[1]
        x_dot =K.batch_dot(x_temp,x_fea)
        x_reshaped = tf.reshape(x_dot,[K.shape(x_dot)[0]*K.shape(x_dot)[1],1])
        ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
        intermed = tf.reduce_sum(tf.multiply(self.uw, ui), axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x_temp)[0], K.shape(x_temp)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x_temp*weights
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        s =input_shape[0]
        return (s[0],s[2])

#Attention =x_temp * softmax(uW*tanh(W_temp*x_temp+W_fea*x_fea+ bw))
#model Att_3
class AttLayer_model_3(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
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

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x_temp)[0], K.shape(x_temp)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x_temp*weights
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        s =input_shape[0]
        return (s[0],s[2])



#Attention =x_temp * softmax(uW*tanh(W_temp*x_temp.*W_fea*x_fea)+ bw)
#model Att_3
class AttLayer_model_4(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim =attention_size
        super(AttLayer_model_4,self).__init__(**kwargs)

    def build(self, input_shape):
        s_temp =input_shape[0]
        self.W_temp = self.add_weight(shape=(s_temp[-1], self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_temp')
        self.W_fea = self.add_weight(shape=(1,self.hidden_dim), initializer = glorot_uniform(seed=SEED), trainable=True,name='W_fea')
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True,name='bw')
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = glorot_uniform(seed=SEED), trainable=True,name='uw')
        self.trainable_weights = [self.W_temp,self.W_fea, self.bw, self.uw]
        super(AttLayer_model_4,self).build(input_shape)

    def call(self, x, mask=None):
        x_temp =x[0]  #(none,50,5)
        x_reshaped = tf.reshape(x_temp, [K.shape(x_temp)[0]*K.shape(x_temp)[1], K.shape(x_temp)[-1]]) #(none*50,5)
        temp =K.dot(x_reshaped, self.W_temp) #(none*50,5)

        x_fea =x[1] #(none ,50)
        x_fea =tf.reshape(x_fea,[K.shape(x_fea)[0]*K.shape(x_fea)[1],1]) #(none*50,1)
        fea = K.dot(x_fea, self.W_fea) #(none*50,5)
        intermed =K.tanh(K.batch_dot(temp,fea,axes=1))
        intermed = tf.reduce_sum(tf.multiply(self.uw, intermed)+self.bw, axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x_temp)[0], K.shape(x_temp)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x_temp*weights
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        s =input_shape[0]
        return (s[0],s[2])



#get the max output in lstm all h_state output
class MaxoverTime(Layer):
    def __init__(self,**kwargs):
        super(MaxoverTime,self).__init__(**kwargs)

    def call(self,x,mask=None):
        return K.max(x,axis=1)

    def compute_output_shape(self,input_shape):
        return input_shape[0],input_shape[-1]

class MeanoverTime(Layer):
    def __init__(self,**kwargs):
        super(MeanoverTime,self).__init__(**kwargs)

    def call(self,x,mask=None):
        return K.mean(x,axis=1)

    def compute_output_shape(self,input_shape):
        return input_shape[0],input_shape[-1]


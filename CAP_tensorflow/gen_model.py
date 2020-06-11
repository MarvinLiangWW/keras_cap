import tensorflow as tf
class Train_Model():
    def __init__(self,user_num,lamda,feature_dim=10,emd_dim=16,init_delta=0.05,lr =0.001):
        self.emd_dim =emd_dim
        self.user_num =user_num
        self.init_delta =init_delta
        self.learning_rate =lr
        self.lamda =lamda
        self.feature_dim =feature_dim

        #input
        self.feature = tf.placeholder(tf.float32, [None, 10], name='feature')
        self.id_embedding =tf.placeholder(tf.int32,name='id_embedding')
        self.label =tf.placeholder(tf.float32,[None,2],name ='label')

        #variable
        self.user_embeddings = tf.Variable(
            tf.random_uniform([self.user_num, self.emd_dim], minval=-self.init_delta, maxval=self.init_delta,
                              dtype=tf.float32, seed=133,name='user_embedding'))
        # self.user_bias = tf.Variable(tf.zeros([self.user_num]))
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.id_embedding, name='u_embedding')
        self.W = tf.Variable(tf.random_uniform(shape=[feature_dim+emd_dim, 2], seed=133, minval=-1.0, maxval=1.0), name='weight', )
        self.b = tf.Variable(tf.random_normal(shape=[2], seed=133, mean=0, stddev=1), name='bias', )

        self.concat_feature_embedding =tf.concat([self.feature,self.u_embedding],axis=1)
        self.y_predict = tf.matmul(self.concat_feature_embedding, self.W) + self.b

        #evaluation
        self.correct_prediction = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.loss =tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.y_predict) + self.lamda * (
            tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b) + tf.nn.l2_loss(self.u_embedding))
        self.adam_opt =tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updates = self.adam_opt.minimize(self.loss)





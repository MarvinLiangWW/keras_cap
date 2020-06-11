from cap_util import get_train_test_data_X,read_csv,read_dat,dense_to_one_hot,get_train_test_data_Y,get_train_test_data_temp
from cap_util import get_train_test_data,mse
from gen_model import *
from scipy.stats import mode
from batch import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from keras.layers import merge
from matplotlib.pyplot import savefig,plot,legend,show,title,subplot,figure
import tensorflow as tf
import keras.backend as K
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _state_size_with_prefix
import os
seed =133
np.random.seed(seed)
tf.set_random_seed(seed)
batch_size = 8
min_after_dequeue = 1000
training_epoches=200
train_example =490
test_example =209
all_example =699
max_len =50
alpha =0.1
#组合样例的队列里最多可以存储的样例个数。队列如果太大，则会占用很多的内存资源，太少，那么出队操作可能会因为没有数据而被阻碍(block)
#从而导致训练效率降低，一般队列大小与batch的大小相关，如果设置了min_after_dequeue那么ca=min+3*bat
capacity = min_after_dequeue + 3 * batch_size
lstm_unit =16

def get_session(memory_rate = 0.9, gpus = '0'):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = memory_rate,
            visible_device_list = gpus,
            #allow_growth = True,
            )

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




def para_prediction(data_version):

    X = read_csv('cap_feature.csv')
    y_ = read_csv('number_category.csv')
    nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test = get_train_test_data(X, y_, nb_x_train, nb_x_test, )
    y_train =dense_to_one_hot(y_train,2)
    y_test =dense_to_one_hot(y_test,2)
    print(x_train.shape)
    print(y_train.shape)

    train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                          batch_size=batch_size,
                                                          min_after_dequeue=min_after_dequeue,
                                                          allow_smaller_final_batch=True,
                                                          capacity=capacity,
                                                          enqueue_many=True,
                                                          seed=133)

    x_input = tf.placeholder(tf.float32, [None, 10],name='feature_input')
    y_label = tf.placeholder(tf.float32,[None,2],name='CAP_category')

    # W = tf.Variable(tf.random_normal(shape=[10,2],seed=133,mean=0,stddev=1), name='weight',)
    W = tf.Variable(tf.random_uniform(shape=[10,2],seed=133,minval=-1.0,maxval=1.0), name='weight',)
    b = tf.Variable(tf.random_normal(shape=[2],seed=133,mean=0,stddev=1), name='bias',)
    y_predict = tf.matmul(x_input, W) + b
    #dropout
    # keep_prob = tf.placeholder(tf.float32,name='dropout')
    # y_predict = tf.nn.dropout(y_predict, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    cross_entropy = cross_entropy + tf.contrib.layers.l2_regularizer(0.01)(W)+tf.contrib.layers.l2_regularizer(0.01)(b)

    #evaluation
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train_step = tf.train.AdamOptimizer(learning_rate=0.0006,).minimize(cross_entropy)
    train_step =tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    test_acc_list=[]
    training_loss_list=[]
    testing_loss_list=[]
    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 循环的训练神经网络。
        for k in range(training_epoches):
            training_steps =int(train_example/batch_size)

            for i in range(training_steps):
                x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
                # print(y_batch)
                # print(sess.run(y_predict,feed_dict={x_input: x_batch, y_label: y_batch}))
                sess.run(train_step, feed_dict={x_input: x_batch, y_label: y_batch})
                if i % 100 == 0:
                    print("After %d training step(s), loss is %g " % (
                    i, sess.run(cross_entropy, feed_dict={x_input: x_train, y_label: y_train,})))
                    print("accuracy:", sess.run(accuracy, feed_dict={x_input: x_train, y_label: y_train,}))
            #training_loss
            training_loss =sess.run(cross_entropy,feed_dict={x_input:x_train,y_label:y_train})
            training_loss_list.append(training_loss)
            #testing_loss
            testing_loss =sess.run(cross_entropy,feed_dict={x_input:x_test,y_label:y_test})
            testing_loss_list.append(testing_loss)
            # calculate test accuracy
            test_accuracy = sess.run(accuracy, feed_dict={x_input: x_test, y_label: y_test,})
            print("testing accuracy: {}".format(test_accuracy))
            test_acc_list.append(test_accuracy)

        coord.request_stop()
        coord.join(threads)
    plot(range(0,training_epoches),test_acc_list,label='test_acc_%d'%(data_version))
    plot(range(0,training_epoches),training_loss_list,label='train_loss_%d'%(data_version))
    plot(range(0,training_epoches),testing_loss_list,label='test_loss_%d'%(data_version))
    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d'%(data_version))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d'%(data_version))
    legend()


def para_prediction_batch(data_version):
    X = read_csv('cap_feature.csv')
    y_ = read_csv('number_category.csv')
    nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    x_train, x_test, y_train, y_test = get_train_test_data(X, y_, nb_x_train, nb_x_test, )
    y_train =dense_to_one_hot(y_train,2)
    y_test =dense_to_one_hot(y_test,2)
    print(x_train.shape)
    print(y_train.shape)

    Data_train = BatchGenerator(X=x_train, y=y_train, shuffle=True)


    x_input = tf.placeholder(tf.float32, [None, 10],name='feature_input')

    y_label = tf.placeholder(tf.float32,[None,2],name='CAP_category')


    W = tf.Variable(tf.random_normal(shape=[10,2],seed=seed,mean=0,stddev=1), name='weight',)
    # W = tf.Variable(tf.VarianceScaling(scale=1.0,mode="fan_avg",distribution="uniform",seed=133))
    # W = tf.Variable(tf.random_uniform(shape=[10,2],seed=133,minval=-1.0,maxval=1.0), name='weight',)
    b = tf.Variable(tf.random_normal(shape=[2],seed=seed,mean=0,stddev=1), name='bias',)
    y_predict = tf.matmul(x_input, W) + b
    #dropout
    # keep_prob = tf.placeholder(tf.float32,name='dropout')
    # y_predict = tf.nn.dropout(y_predict, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    cross_entropy = cross_entropy + tf.contrib.layers.l2_regularizer(0.01)(W)+tf.contrib.layers.l2_regularizer(0.01)(b)

    #evaluation
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train_step = tf.train.AdamOptimizer(learning_rate=0.0006,).minimize(cross_entropy)
    train_step =tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    test_acc_list=[]
    training_loss_list=[]
    testing_loss_list=[]
    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 循环的训练神经网络。
        for k in range(training_epoches):
            training_steps =int(train_example/batch_size)

            for i in range(training_steps):
                x_batch, y_batch = Data_train.next_batch(batch_size=batch_size)
                # print(y_batch)
                # print(sess.run(y_predict,feed_dict={x_input: x_batch, y_label: y_batch}))
                sess.run(train_step, feed_dict={x_input: x_batch, y_label: y_batch})
                if i % 100 == 0:
                    print("After %d training step(s), loss is %g " % (
                    i, sess.run(cross_entropy, feed_dict={x_input: x_train, y_label: y_train,})))
                    print("accuracy:", sess.run(accuracy, feed_dict={x_input: x_train, y_label: y_train,}))
            #training_loss
            training_loss =sess.run(cross_entropy,feed_dict={x_input:x_train,y_label:y_train})
            training_loss_list.append(training_loss)
            #testing_loss
            testing_loss =sess.run(cross_entropy,feed_dict={x_input:x_test,y_label:y_test})
            testing_loss_list.append(testing_loss)
            # calculate test accuracy
            test_accuracy = sess.run(accuracy, feed_dict={x_input: x_test, y_label: y_test,})
            print("testing accuracy: {}".format(test_accuracy))
            test_acc_list.append(test_accuracy)

        coord.request_stop()
        coord.join(threads)
    plot(range(0,training_epoches),test_acc_list,label='test_acc_%d'%(data_version))
    plot(range(0,training_epoches),training_loss_list,label='train_loss_%d'%(data_version))
    plot(range(0,training_epoches),testing_loss_list,label='test_loss_%d'%(data_version))
    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d'%(data_version))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d'%(data_version))
    legend()


def prediction_(data_version):

    X = read_csv('cap_feature.csv')
    y_ = read_csv('number_category.csv')
    nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    x_line_num =read_csv('line_number.csv')
    x_train, x_test, y_train, y_test,x_train_lineNum,x_test_lineNum = get_merge_train_test_data(X,x_line_num, y_, nb_x_train, nb_x_test, )
    y_train =dense_to_one_hot(y_train,2)
    y_test =dense_to_one_hot(y_test,2)
    print(x_train.shape)
    print(x_train_lineNum.shape)
    print(y_train.shape)

    train_x_batch,train_x_lineNum_batch ,train_y_batch = tf.train.shuffle_batch([x_train, x_train_lineNum,y_train],
                                                          batch_size=batch_size,
                                                          min_after_dequeue=min_after_dequeue,
                                                          allow_smaller_final_batch=True,
                                                          capacity=capacity,
                                                          enqueue_many=True,
                                                          seed=133)
    model =Train_Model(user_num=700,lamda=0.001,)



    test_acc_list=[]
    testing_loss_list=[]
    training_loss_list = []
    training_accurary_list=[]
    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 循环的训练神经网络。
        for k in range(training_epoches):
            training_steps =int(train_example/batch_size)

            for i in range(training_steps):
                x_batch,x_lineNum_batch, y_batch = sess.run([train_x_batch,train_x_lineNum_batch, train_y_batch])
                sess.run(model.updates, feed_dict={model.feature: x_batch, model.id_embedding:x_lineNum_batch,model.label: y_batch})
                if i % 100 == 0:
                    print("After %d training step(s), loss is %g " % (
                    i, sess.run(model.loss, feed_dict={model.feature: x_batch, model.id_embedding:x_lineNum_batch,model.label: y_batch})))
                    print("accuracy:", sess.run(model.accuracy, feed_dict={model.feature: x_batch, model.id_embedding:x_lineNum_batch,model.label: y_batch}))
            #training_loss
            training_loss =sess.run(model.loss,feed_dict={model.feature: x_train, model.id_embedding:x_train_lineNum,model.label: y_train})
            training_loss_list.append(training_loss)
            #training_accurary
            training_accurary =sess.run(model.accuracy,feed_dict={model.feature:x_train,model.id_embedding:x_train_lineNum,model.label: y_train})
            training_accurary_list.append(training_accurary)
            # #testing_loss
            # testing_loss =sess.run(cross_entropy,feed_dict={x_input:x_test,y_label:y_test})
            # testing_loss_list.append(testing_loss)
            # # calculate test accuracy
            # test_accuracy = sess.run(accuracy, feed_dict={x_input: x_test, y_label: y_test,})
            # print("testing accuracy: {}".format(test_accuracy))
            # test_acc_list.append(test_accuracy)

        coord.request_stop()
        coord.join(threads)
    plot(range(0,training_epoches),training_accurary_list,label='train_acc_%d'%(data_version))
    plot(range(0,training_epoches),training_loss_list,label='train_loss_%d'%(data_version))
    # plot(range(0,training_epoches),testing_loss_list,label='test_loss_%d'%(data_version))
    # plot(range(0, training_epoches), test_acc_list, label='test_acc_%d' % (data_version))
    # test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d'%(data_version))
    # print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    # print(test_acc_list)
    # print(mode(test_acc_list))
    title('data_version_%d'%(data_version))
    legend()


def merge_prediction_batch(data_version):
    X = read_csv('cap_feature.csv')
    temp =read_csv('temperature_minmax.csv')
    y_ = read_csv('number_category.csv')
    nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    x_train, x_test= get_train_test_data_X(X, nb_x_train, nb_x_test, )
    temp_train, temp_test = get_train_test_data_temp(temp, nb_x_train, nb_x_test )
    temp_train = np.reshape(temp_train, (temp_train.shape[0], temp_train.shape[1], 1))
    temp_test = np.reshape(temp_test, (temp_test.shape[0], temp_test.shape[1], 1))
    y_train, y_test = get_train_test_data_Y(y_, nb_x_train, nb_x_test, )
    y_train = dense_to_one_hot(y_train, 2)
    y_test = dense_to_one_hot(y_test, 2)
    Data_train = BatchGeneratorFeatureTemp(X=x_train,temp =temp_train,y=y_train, shuffle=True)

    x_input = tf.placeholder(tf.float32, [None, 10],name='feature_input')
    temp_input =tf.placeholder(tf.float32,[None,50,1],name='temp_input')
    y_label = tf.placeholder(tf.float32,[None,2],name='CAP_category')

    weights ={
        'w_input':tf.Variable(tf.random_normal(shape=[10,8],name='w_input')),
        'w_out':tf.Variable(tf.random_normal(shape=(16,2),name='w_out'))
    }
    bias ={
        'b_input':tf.Variable(tf.random_normal(shape=[8,],name ='b_input')),
        'b_out':tf.Variable(tf.random_normal(shape=[2,],name='b_out'))
    }

    x_inter=tf.matmul(x_input, weights['w_input']) + bias['b_input']

    #lstm_cell with dropout
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=8, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)

    initial_state =lstm_cell.zero_state(batch_size, tf.float32)


    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,dtype=tf.float32,inputs=temp_input,initial_state=initial_state)
    output = tf.transpose(outputs, [1, 0, 2])
    merge_tensor =tf.concat([output[-1],x_inter],axis=1)
    y_predict =tf.matmul(merge_tensor, weights['w_out']) + bias['b_out']

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    # cross_entropy = cross_entropy + tf.contrib.layers.l2_regularizer(0.01)(W)+tf.contrib.layers.l2_regularizer(0.01)(b)

    #evaluation
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.AdamOptimizer(learning_rate=0.0006,).minimize(cross_entropy)
    # train_step =tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    test_acc_list=[]
    training_loss_list=[]
    testing_loss_list=[]
    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()
        # 循环的训练神经网络。
        for k in range(training_epoches):
            training_steps =int(num_example/batch_size)

            for i in range(training_steps):
                x_batch,temp_batch, y_batch = Data_train.next_batch(batch_size=batch_size)
                temp_batch =np.reshape(temp_batch,(temp_batch.shape[0],temp_batch.shape[1],1))
                # print(y_batch)
                # print(sess.run(y_predict,feed_dict={x_input: x_batch, y_label: y_batch}))
                sess.run(train_step, feed_dict={x_input: x_batch,temp_input:temp_batch, y_label: y_batch})
                if i % 100 == 0:
                    # print("After %d training step(s), loss is %g " % (
                    # i, sess.run(cross_entropy, feed_dict={x_input: x_train,temp_input:temp_train, y_label: y_train,})))
                    # print("accuracy:", sess.run(accuracy, feed_dict={x_input: x_train, temp_input:temp_train,y_label: y_train,}))
                    print("After %d training step(s), loss is %g " % (i, cross_entropy.eval(feed_dict={x_input: x_train,temp_input:temp_train, y_label: y_train,})))
                    print("accuracy:", accuracy.eval(feed_dict={x_input: x_train, temp_input:temp_train,y_label: y_train,}))
            #training_loss
            training_loss =sess.run(cross_entropy,feed_dict={x_input:x_train,temp_input:temp_train,y_label:y_train})
            training_loss_list.append(training_loss)
            #testing_loss
            testing_loss =sess.run(cross_entropy,feed_dict={x_input:x_test,temp_input:temp_test,y_label:y_test})
            testing_loss_list.append(testing_loss)
            # calculate test accuracy
            test_accuracy = sess.run(accuracy, feed_dict={x_input: x_test,temp_input:temp_test, y_label: y_test,})
            print("testing accuracy: {}".format(test_accuracy))
            test_acc_list.append(test_accuracy)

    plot(range(0,training_epoches),test_acc_list,label='test_acc_%d'%(data_version))
    plot(range(0,training_epoches),training_loss_list,label='train_loss_%d'%(data_version))
    plot(range(0,training_epoches),testing_loss_list,label='test_loss_%d'%(data_version))
    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d'%(data_version))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d'%(data_version))
    legend()


def transfer_learning_fail(data_version):
    X = read_csv('cap_feature.csv')
    lineNum =read_csv('line_number.csv')
    temp_before = read_csv('front_1_10_temp.csv')
    temp_after = read_csv('front_2_11_temp.csv')
    y_ = read_csv('number_category.csv')
    nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    x_train, x_test = get_train_test_data_X(X, nb_x_train, nb_x_test, )
    lineNum_train,lineNum_test =get_train_test_data_X(lineNum,nb_x_train,nb_x_test)
    temp_before_train, temp_before_test = get_train_test_data_X(temp_before, nb_x_train, nb_x_test)
    temp_after_train, temp_after_test = get_train_test_data_X(temp_after, nb_x_train, nb_x_test)

    temp_before_train = np.reshape(temp_before_train, (temp_before_train.shape[0], temp_before_train.shape[1], 1))
    temp_before_test = np.reshape(temp_before_test, (temp_before_test.shape[0], temp_before_test.shape[1], 1))
    temp_after_train = np.reshape(temp_after_train, (temp_after_train.shape[0], temp_after_train.shape[1], 1))
    temp_after_test = np.reshape(temp_after_test, (temp_after_test.shape[0], temp_after_test.shape[1], 1))

    y_train, y_test = get_train_test_data_Y(y_, nb_x_train, nb_x_test, )
    y_train = dense_to_one_hot(y_train, 2)
    y_test = dense_to_one_hot(y_test, 2)

    Data_train = BatchGeneratorTransferLearning(X=x_train, lineNum=lineNum_train,temp_before =temp_before_train,temp_after=temp_after_train, y=y_train, shuffle=True)
    Data_test = BatchGeneratorTransferLearning(X=x_test, lineNum=lineNum_test,temp_before =temp_before_test,temp_after=temp_after_test, y=y_test, shuffle=False)

    x_input = tf.placeholder(tf.float32, [None, 10], name='feature_input')
    lineNum_input =tf.placeholder(tf.int32,[None,1],name='lineNum_input')
    temp_before_input = tf.placeholder(tf.float32, [None, 10, 1], name='temp_before_input')
    y_label = tf.placeholder(tf.float32, [None, 2], name='CAP_category')
    y_temp_output =tf.placeholder(tf.float32,[None,10,1],name='y_temp_output')
    # alpha =tf.placeholder(tf.float32,[1],name='alpha')

    embeddings = tf.Variable(tf.random_uniform([all_example+1, 16], -1.0, 1.0,seed=seed),)
    embeded = tf.nn.embedding_lookup(embeddings, ids=lineNum_input)

    weights = {
        'w_input': tf.Variable(tf.random_normal(shape=(10, 8), name='w_input',seed=seed)),
        'w_out': tf.Variable(tf.random_normal(shape=(24, 2), name='w_out',seed=seed)),
        'w_temp':tf.Variable(tf.random_normal(shape=(16,1),name ='w_temp',seed=seed))
    }
    bias = {
        'b_input': tf.Variable(tf.random_normal(shape=(8, ), name='b_input',seed=seed)),
        'b_out': tf.Variable(tf.random_normal(shape=(2, ), name='b_out',seed=seed)),
        'b_temp': tf.Variable(tf.random_normal(shape=(1, ), name='b_temp',seed=seed))
    }

    x_inter = tf.matmul(x_input, weights['w_input']) + bias['b_input']

    embed =tf.reshape(embeded,shape =(batch_size,16),name ='EmbeddingReshape')
    x_merge =tf.concat([x_inter,embed],axis=1)
    y_predict =tf.matmul(x_merge,weights['w_out'])+bias['b_out']


    # lstm_cell with dropout
    with tf.variable_scope('lstmcell',reuse=True):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16, state_is_tuple=True,reuse= tf.get_variable_scope().reuse)
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16, state_is_tuple=True,reuse= True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5,)

    # initial_state =lstm_cell.zero_state(batch_size,np.float32)
    # h_state =tf.Variable(tf.zeros(shape=(batch_size,16,),name='h_state'))
    h_state =embed
    c_state =tf.Variable(tf.zeros(shape=(batch_size,16,),name='c_state',),trainable=False)
    state_tuple = LSTMStateTuple(c_state, h_state)

    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=temp_before_input,
                                             initial_state=state_tuple,)
    output =tf.reshape(outputs,shape=(K.shape(outputs)[0]*K.shape(outputs)[1],K.shape(outputs)[2]))
    y_temp_predict =tf.matmul(output,weights['w_temp'])+bias['b_temp']
    y_temp_predict =tf.reshape(y_temp_predict,shape=(K.shape(outputs)[0],K.shape(outputs)[1],1))


    #loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    mse =tf.reduce_mean(tf.square(tf.subtract(y_temp_predict, y_temp_output)))
    loss =cross_entropy*alpha+(1-alpha)*mse

    train_step = tf.train.AdamOptimizer(learning_rate=0.0006, ).minimize(loss)

    # train_step =tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)




    #training step 2 model

    test_h_state =tf.Variable(tf.zeros(shape=(batch_size,16,),name='test_h_state',),trainable=True)
    test_c_state =tf.Variable(tf.zeros(shape=(batch_size,16,),name='test_c_state',),trainable=False)
    test_state_tuple = LSTMStateTuple(test_c_state, test_h_state)
    outputs_two, last_states_two = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=temp_before_input,
                                             initial_state=test_state_tuple,)
    output_two = tf.reshape(outputs_two, shape=(K.shape(outputs_two)[0] * K.shape(outputs_two)[1], K.shape(outputs_two)[2]))
    h_state_predict = tf.matmul(output_two, weights['w_temp']) + bias['b_temp']
    h_state_predict = tf.reshape(h_state_predict, shape=(K.shape(outputs)[0], K.shape(outputs)[1], 1))

    mse_step_two =tf.reduce_mean(tf.square(tf.subtract(h_state_predict, y_temp_output)))
    train_step_2 =tf.train.AdamOptimizer(learning_rate=0.0005).minimize(mse_step_two)

    test_fea_merge = tf.concat([x_inter, test_h_state], axis=1)
    test_y_predict = tf.matmul(test_fea_merge, weights['w_out']) + bias['b_out']
    # evaluation
    correct_prediction = tf.equal(tf.argmax(test_y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #training ...
    train_acc_list=[]
    test_acc_list = []
    training_loss_list = []
    testing_loss_list = []
    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()
        # 循环的训练神经网络。
        for k in range(training_epoches):
            #training step 1
            training_steps = int(train_example / batch_size)
            for i in range(training_steps):
                x_batch, lineNum_batch, temp_before_batch, temp_after_batch, y_batch = Data_train.next_batch(
                    batch_size=batch_size)
                feed_train_batch_dicts = {x_input: x_batch, lineNum_input: lineNum_batch,
                                          temp_before_input: temp_before_batch, y_temp_output: temp_after_batch,
                                          y_label: y_batch, }
                sess.run(train_step, feed_dict=feed_train_batch_dicts)



            #training step 2
            test_accuracy=0.0
            training_step_2 =int(test_example/batch_size)
            for _ in range(training_step_2):
                x_test_batch, lineNum_test_batch, temp_before_test_batch, temp_after_test_batch, y_test_batch = Data_test.next_batch(
                    batch_size=batch_size)
                feed_test_2_batch_dicts = {temp_before_input: temp_before_test_batch,
                                         y_temp_output: temp_after_test_batch }
                sess.run(train_step_2,feed_dict=feed_test_2_batch_dicts)
                acc =sess.run(accuracy,feed_dict={x_input:x_test_batch,lineNum_input:lineNum_test_batch,temp_before_input:temp_before_test_batch,
                                                  y_temp_output:temp_after_test_batch,y_label:y_test_batch})
                test_accuracy+=acc
            print("testing accuracy: {}".format(test_accuracy/training_steps))
            test_acc_list.append(test_accuracy/training_steps)

    plot(range(0, training_epoches), test_acc_list, label='test_acc_%d' % (data_version))
    plot(range(0, training_epoches), train_acc_list, label='train_acc_%d' % (data_version))
    plot(range(0, training_epoches), training_loss_list, label='train_loss_%d' % (data_version))
    plot(range(0, training_epoches), testing_loss_list, label='test_loss_%d' % (data_version))
    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d' % (data_version))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d' % (data_version))
    legend()


def get_initial_cell_state(cell, initializer, batch_size, dtype,h_state):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        c_state =zero_state_initializer(_state_size_with_prefix(state_size_flat[0]),batch_size,dtype,0)
        # h_state =initializer(_state_size_with_prefix(state_size_flat[1]),batch_size,dtype,1)
        h_state =h_state
        init_state_flat = [c_state,h_state]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)
    return init_state

def zero_state_initializer(shape, batch_size, dtype, index):
    z = tf.zeros(tf.stack(_state_size_with_prefix(shape, [batch_size])), dtype)
    z.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return z

def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()
        print(args)

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        print('var',var)
        var = tf.expand_dims(var, 0)
        print(var)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        print('tile',var)
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        print('final var:',var)
        return var

    return variable_state_initializer

def transfer_learning_(data_version,b=1):
    X = read_csv('cap_feature.csv')
    lineNum = read_csv('line_number.csv')
    temp_before = read_csv('front_1_10_temp.csv')
    temp_after = read_csv('front_2_11_temp.csv')
    y_ = read_csv('number_category.csv')
    # nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    # nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    nb_x_train =read_dat('nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_dat('nb_test_cv%d.dat'%(data_version))

    x_train, x_test = get_train_test_data_X(X, nb_x_train, nb_x_test, )
    lineNum_train, lineNum_test = get_train_test_data_X(lineNum, nb_x_train, nb_x_test)
    temp_before_train, temp_before_test = get_train_test_data_X(temp_before, nb_x_train, nb_x_test)
    temp_after_train, temp_after_test = get_train_test_data_X(temp_after, nb_x_train, nb_x_test)

    temp_before_train = np.reshape(temp_before_train, (temp_before_train.shape[0], temp_before_train.shape[1], 1))
    temp_before_test = np.reshape(temp_before_test, (temp_before_test.shape[0], temp_before_test.shape[1], 1))
    temp_after_train = np.reshape(temp_after_train, (temp_after_train.shape[0], temp_after_train.shape[1], 1))
    temp_after_test = np.reshape(temp_after_test, (temp_after_test.shape[0], temp_after_test.shape[1], 1))

    y_train, y_test = get_train_test_data_Y(y_, nb_x_train, nb_x_test, )
    y_train = dense_to_one_hot(y_train, 2)
    y_test = dense_to_one_hot(y_test, 2)

    Data_train = BatchGeneratorTransferLearning(X=x_train, lineNum=lineNum_train, temp_before=temp_before_train,
                                                temp_after=temp_after_train, y=y_train, shuffle=True)
    Data_test = BatchGeneratorTransferLearning(X=x_test, lineNum=lineNum_test, temp_before=temp_before_test,
                                               temp_after=temp_after_test, y=y_test, shuffle=False)

    x_input = tf.placeholder(tf.float32, [None, 10], name='feature_input')
    lineNum_input = tf.placeholder(tf.int32, [None, 1], name='lineNum_input')
    temp_before_input = tf.placeholder(tf.float32, [None, 10, 1], name='temp_before_input')
    y_label = tf.placeholder(tf.float32, [None, 2], name='CAP_category')
    y_temp_output = tf.placeholder(tf.float32, [None, 10, 1], name='y_temp_output')
    dropout = tf.placeholder(tf.float32)


    embeddings = tf.Variable(tf.random_uniform([all_example + 1, 16], -1.0, 1.0, seed=seed),trainable=True,name='embeddings')
    embeded = tf.nn.embedding_lookup(embeddings, ids=lineNum_input)
    embeded =tf.nn.dropout(embeded,keep_prob=dropout,seed=seed)


    weights = {
        'w_input': tf.Variable(tf.random_normal(shape=(10, 8), name='w_input', seed=seed)),
        'w_out': tf.Variable(tf.random_normal(shape=(24, 2), name='w_out', seed=seed)),
        'w_temp': tf.Variable(tf.random_normal(shape=(16, 1), name='w_temp', seed=seed)),
        'w_lstmcell':tf.Variable(tf.random_normal(shape=(17,64),name='w_lstmcell',seed=seed)),
    }
    bias = {
        'b_input': tf.Variable(tf.random_normal(shape=(8,), name='b_input', seed=seed)),
        'b_out': tf.Variable(tf.random_normal(shape=(2,), name='b_out', seed=seed)),
        'b_temp': tf.Variable(tf.random_normal(shape=(1,), name='b_temp', seed=seed)),
        'b_lstmcell':tf.Variable(tf.random_normal(shape=(64,),name='b_lstmcell',seed=seed))
    }


    x_inter = tf.matmul(x_input, weights['w_input']) + bias['b_input']
    x_inter =tf.nn.dropout(x_inter,dropout,seed=seed)

    embed = tf.reshape(embeded, shape=(batch_size, 16), name='EmbeddingReshape')
    x_merge = tf.concat([x_inter, embed], axis=1)
    y_predict = tf.matmul(x_merge, weights['w_out']) + bias['b_out']
    # y_predict =tf.nn.dropout(y_predict,keep_prob=0.9,seed=seed)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16, state_is_tuple=True)
    # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5,)
    # don't know why
    # add the top line code won't get the same result every training
    # maybe the random sequence is changing


    # h_state = embed
    # c_state = tf.Variable(tf.zeros(shape=(batch_size, 16,), name='c_state', ), trainable=True)
    # state_tuple = LSTMStateTuple(c_state, h_state)

    initializer = make_variable_state_initializer()
    init_state = get_initial_cell_state(lstm_cell, initializer, batch_size, tf.float32,h_state=embed)


    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=temp_before_input,
                                             initial_state=init_state, )
    output = tf.reshape(outputs, shape=(K.shape(outputs)[0] * K.shape(outputs)[1], K.shape(outputs)[2]))
    y_temp_predict = tf.matmul(output, weights['w_temp']) + bias['b_temp']
    y_temp_predict = tf.reshape(y_temp_predict, shape=(K.shape(outputs)[0], K.shape(outputs)[1], 1))

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    mse = tf.reduce_mean(tf.square(tf.subtract(y_temp_predict, y_temp_output)))
    loss = cross_entropy * alpha + (1 - alpha) * mse

    train_step = tf.train.AdamOptimizer(learning_rate=0.0006, ).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    vars_name =[v.name for v in tf.trainable_variables()]
    print(vars_name)

    lstm_cell_weight =[v for v in tf.trainable_variables() if v.name=='rnn/basic_lstm_cell/weights:0']
    print(lstm_cell_weight[0].shape)

    lstm_cell_bias =[v for v in tf.trainable_variables() if v.name=='rnn/basic_lstm_cell/biases:0']
    print(lstm_cell_bias[0].shape)


    optimizer = tf.train.AdamOptimizer(learning_rate=0.0006, )
    tvars = [v for v in tf.trainable_variables() if v.name =='embeddings:0']
    gradients = tf.gradients(mse, tvars)
    gradients, _ = tf.clip_by_global_norm(gradients,1.0)

    train_step_2 =optimizer.apply_gradients(zip(gradients, tvars))

    # training ...
    test_acc_list = []
    train_acc_list=[]
    training_loss_list=[]
    test_loss_list=[]
    # test_acc_step_list = []
    # test_loss_step_list = []

    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()

        lstm_cell_weight[0].load(sess.run(weights['w_lstmcell']))
        lstm_cell_bias[0].load(sess.run(bias['b_lstmcell']))


        # 循环的训练神经网络。
        for k in range(training_epoches):
            # training step 1
            training_loss =0.0
            training_accuracy =0.0
            cross_entropy_loss =0.0
            training_steps = int(train_example / batch_size)
            for i in range(training_steps):
                x_batch, lineNum_batch, temp_before_batch, temp_after_batch, y_batch = Data_train.next_batch(
                    batch_size=batch_size)
                feed_train_batch_dicts = {x_input: x_batch, lineNum_input: lineNum_batch,
                                          temp_before_input: temp_before_batch, y_temp_output: temp_after_batch,
                                          y_label: y_batch,dropout:0.5 }

                sess.run(train_step, feed_dict=feed_train_batch_dicts)
                train_batch_loss =sess.run(loss,feed_dict=feed_train_batch_dicts)
                training_loss+=train_batch_loss
                train_batch_cross_entropy =sess.run(cross_entropy,feed_dict=feed_train_batch_dicts)
                cross_entropy_loss+=train_batch_cross_entropy
                train_acc =sess.run(accuracy,feed_dict=feed_train_batch_dicts)
                training_accuracy+=train_acc
            print('train_loss:{}'.format(training_loss/training_steps))
            training_loss_list.append(training_loss/training_steps)
            print('cross_entropy_loss:{}'.format(cross_entropy_loss/training_steps))
            print('train_accuracy:{}'.format(training_accuracy/training_steps))
            train_acc_list.append(training_accuracy/training_steps)
            test_acc_epoch_list=[]
            test_loss_epoch_list=[]
            for _ in range(b):
                # training step 2
                test_accuracy = 0.0
                test_loss =0.0
                training_step_2 = int(test_example / batch_size)
                for _ in range(training_step_2):
                    x_test_batch, lineNum_test_batch, temp_before_test_batch, temp_after_test_batch, y_test_batch = Data_test.next_batch(
                        batch_size=batch_size)
                    feed_test_batch_dicts = {x_input: x_test_batch, lineNum_input: lineNum_test_batch,
                                                        temp_before_input: temp_before_test_batch,
                                                        y_temp_output: temp_after_test_batch, y_label: y_test_batch,dropout:1.0}
                    sess.run(train_step_2, feed_dict={temp_before_input: temp_before_test_batch,
                                               lineNum_input: lineNum_test_batch,
                                               y_temp_output: temp_after_test_batch,dropout:0.5})
                    acc = sess.run(accuracy, feed_dict=feed_test_batch_dicts)
                    test_accuracy += acc
                    test_batch_loss =sess.run(loss,feed_dict=feed_test_batch_dicts)
                    test_loss+=test_batch_loss
                print("testing accuracy: {}".format(test_accuracy / training_step_2))
                test_acc_epoch_list.append(test_accuracy/training_step_2)
                test_loss_epoch_list.append(test_loss/training_step_2)

                # test_acc_step_list.append(test_accuracy/training_step_2)
                # test_loss_step_list.append(test_loss/training_step_2)
            test_acc_list.append(test_acc_epoch_list[-1])
            test_loss_list.append(np.mean(test_loss_epoch_list))
    figure()
    plot(range(0, training_epoches), test_acc_list, label='test_acc')
    plot(range(0, training_epoches), training_loss_list, label='train_loss')
    plot(range(0, training_epoches), train_acc_list, label='train_acc')
    plot(range(0, training_epoches), test_loss_list, label='test_loss')
    # figure()
    # plot(range(0,training_epoches*30),test_acc_step_list,label='test_acc_step')
    # plot(range(0,training_epoches*30),test_loss_step_list,label='test_loss_step')

    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d' % (data_version))
    print("top-10 mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print("top-50 mean: %.3f" % np.mean(np.array(test_acc_list[:50])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d' % (data_version))
    legend()

def transfer_learning_temp(data_version,b=1):
    # X = read_csv('cap_feature.csv')
    lineNum = read_csv('line_number.csv')
    temp_before = read_csv('front_1_10_temp.csv')
    temp_after = read_csv('front_2_11_temp.csv')
    y_ = read_csv('number_category.csv')
    # nb_x_train = read_dat('nb_x_train_%d.dat' % (data_version))
    # nb_x_test = read_dat('nb_x_test_%d.dat' % (data_version))
    nb_x_train =read_dat('nb_train_cv%d.dat'%(data_version))
    nb_x_test =read_dat('nb_test_cv%d.dat'%(data_version))

    # x_train, x_test = get_train_test_data_X(X, nb_x_train, nb_x_test, )
    lineNum_train, lineNum_test = get_train_test_data_X(lineNum, nb_x_train, nb_x_test)
    temp_before_train, temp_before_test = get_train_test_data_X(temp_before, nb_x_train, nb_x_test)
    temp_after_train, temp_after_test = get_train_test_data_X(temp_after, nb_x_train, nb_x_test)

    temp_before_train = np.reshape(temp_before_train, (temp_before_train.shape[0], temp_before_train.shape[1], 1))
    temp_before_test = np.reshape(temp_before_test, (temp_before_test.shape[0], temp_before_test.shape[1], 1))
    temp_after_train = np.reshape(temp_after_train, (temp_after_train.shape[0], temp_after_train.shape[1], 1))
    temp_after_test = np.reshape(temp_after_test, (temp_after_test.shape[0], temp_after_test.shape[1], 1))

    y_train, y_test = get_train_test_data_Y(y_, nb_x_train, nb_x_test, )
    y_train = dense_to_one_hot(y_train, 2)
    y_test = dense_to_one_hot(y_test, 2)

    Data_train = BatchGeneratorTransferLearningTemp( lineNum=lineNum_train, temp_before=temp_before_train,
                                                temp_after=temp_after_train, y=y_train, shuffle=True)
    Data_test = BatchGeneratorTransferLearningTemp(lineNum=lineNum_test, temp_before=temp_before_test,
                                               temp_after=temp_after_test, y=y_test, shuffle=False)

    # x_input = tf.placeholder(tf.float32, [None, 10], name='feature_input')
    lineNum_input = tf.placeholder(tf.int32, [None, 1], name='lineNum_input')
    temp_before_input = tf.placeholder(tf.float32, [None, 10, 1], name='temp_before_input')
    y_label = tf.placeholder(tf.float32, [None, 2], name='CAP_category')
    y_temp_output = tf.placeholder(tf.float32, [None, 10, 1], name='y_temp_output')
    dropout = tf.placeholder(tf.float32)


    embeddings = tf.Variable(tf.random_uniform([all_example + 1, lstm_unit], -1.0, 1.0, seed=seed),trainable=True,name='embeddings')
    embeded = tf.nn.embedding_lookup(embeddings, ids=lineNum_input)
    embeded =tf.nn.dropout(embeded,keep_prob=dropout,seed=seed)


    weights = {
        'w_input': tf.Variable(tf.random_normal(shape=(10, 8), name='w_input', seed=seed)),
        'w_out': tf.Variable(tf.random_normal(shape=(lstm_unit, 2), name='w_out', seed=seed)),
        'w_temp': tf.Variable(tf.random_normal(shape=(lstm_unit, 1), name='w_temp', seed=seed)),
        'w_lstmcell': tf.Variable(tf.random_normal(shape=(17, 64), name='w_lstmcell', seed=seed)),
    }
    bias = {
        'b_input': tf.Variable(tf.random_normal(shape=(8,), name='b_input', seed=seed)),
        'b_out': tf.Variable(tf.random_normal(shape=(2,), name='b_out', seed=seed)),
        'b_temp': tf.Variable(tf.random_normal(shape=(1,), name='b_temp', seed=seed)),
        'b_lstmcell': tf.Variable(tf.random_normal(shape=(64,), name='b_lstmcell', seed=seed))
    }

    # x_inter = tf.matmul(x_input, weights['w_input']) + bias['b_input']
    # x_inter =tf.nn.dropout(x_inter,dropout)

    embed = tf.reshape(embeded, shape=(batch_size, lstm_unit), name='EmbeddingReshape')
    # x_merge = tf.concat([x_inter, embed], axis=1)
    # embed_dro =tf.nn.dropout(embed,keep_prob=dropout,seed =seed)
    y_predict = tf.matmul(embed, weights['w_out']) + bias['b_out']


    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_unit, state_is_tuple=True)
    # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5, )


    # h_state = embed
    # c_state = tf.Variable(tf.zeros(shape=(batch_size, 16,), name='c_state', ), trainable=True)
    # state_tuple = LSTMStateTuple(c_state, h_state)

    initializer = make_variable_state_initializer()
    init_state = get_initial_cell_state(lstm_cell, initializer, batch_size, tf.float32,h_state=embed)

    outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs=temp_before_input,
                                             initial_state=init_state, )
    output = tf.reshape(outputs, shape=(K.shape(outputs)[0] * K.shape(outputs)[1], K.shape(outputs)[2]))
    y_temp_predict = tf.matmul(output, weights['w_temp']) + bias['b_temp']
    y_temp_predict = tf.reshape(y_temp_predict, shape=(K.shape(outputs)[0], K.shape(outputs)[1], 1))

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    mse = tf.reduce_mean(tf.square(tf.subtract(y_temp_predict, y_temp_output)))
    loss = cross_entropy * alpha + (1 - alpha) * mse

    train_step = tf.train.AdamOptimizer(learning_rate=0.0006, ).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    vars_name =[v.name for v in tf.trainable_variables()]
    print(vars_name)

    lstm_cell_weight =[v for v in tf.trainable_variables() if v.name=='rnn/basic_lstm_cell/weights:0']
    print(lstm_cell_weight[0].shape)

    lstm_cell_bias =[v for v in tf.trainable_variables() if v.name=='rnn/basic_lstm_cell/biases:0']
    print(lstm_cell_bias[0].shape)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0006, )
    tvars = [v for v in tf.trainable_variables() if v.name =='embeddings:0']
    gradients = tf.gradients(mse, tvars)
    gradients, _ = tf.clip_by_global_norm(gradients,1.0)

    train_step_2 =optimizer.apply_gradients(zip(gradients, tvars))

    # training ...
    test_acc_list = []
    train_acc_list=[]
    training_loss_list=[]
    test_loss_list=[]

    with tf.Session() as sess:
        print('session')
        tf.global_variables_initializer().run()

        lstm_cell_weight[0].load(sess.run(weights['w_lstmcell']))
        lstm_cell_bias[0].load(sess.run(bias['b_lstmcell']))



        # 循环的训练神经网络。
        for k in range(training_epoches):
            # training step 1
            training_loss =0.0
            training_accuracy =0.0
            cross_entropy_loss =0.0
            training_steps = int(train_example / batch_size)
            for i in range(training_steps):
                lineNum_batch, temp_before_batch, temp_after_batch, y_batch = Data_train.next_batch(
                    batch_size=batch_size)
                feed_train_batch_dicts = {lineNum_input: lineNum_batch,
                                          temp_before_input: temp_before_batch, y_temp_output: temp_after_batch,
                                          y_label: y_batch,dropout:0.5 }
                sess.run(train_step, feed_dict=feed_train_batch_dicts)
                train_batch_loss =sess.run(loss,feed_dict=feed_train_batch_dicts)
                training_loss+=train_batch_loss
                train_batch_cross_entropy =sess.run(cross_entropy,feed_dict=feed_train_batch_dicts)
                cross_entropy_loss+=train_batch_cross_entropy
                train_acc =sess.run(accuracy,feed_dict=feed_train_batch_dicts)
                training_accuracy+=train_acc
            print('train_loss:{}'.format(training_loss/training_steps))
            training_loss_list.append(training_loss/training_steps)
            print('cross_entropy_loss:{}'.format(cross_entropy_loss/training_steps))
            print('train_accuracy:{}'.format(training_accuracy/training_steps))
            train_acc_list.append(training_accuracy/training_steps)
            test_acc_epoch_list=[]
            test_loss_epoch_list=[]
            for _ in range(b):
                # training step 2
                test_accuracy = 0.0
                test_loss =0.0
                training_step_2 = int(test_example / batch_size)
                for _ in range(training_step_2):
                    lineNum_test_batch, temp_before_test_batch, temp_after_test_batch, y_test_batch = Data_test.next_batch(
                        batch_size=batch_size)
                    feed_test_batch_dicts = {lineNum_input: lineNum_test_batch,
                                                        temp_before_input: temp_before_test_batch,
                                                        y_temp_output: temp_after_test_batch, y_label: y_test_batch,dropout:1.0}
                    sess.run(train_step_2, feed_dict={temp_before_input: temp_before_test_batch,
                                               lineNum_input: lineNum_test_batch,
                                               y_temp_output: temp_after_test_batch,dropout:0.5})
                    acc = sess.run(accuracy, feed_dict=feed_test_batch_dicts)
                    test_accuracy += acc
                    test_batch_loss =sess.run(loss,feed_dict=feed_test_batch_dicts)
                    test_loss+=test_batch_loss
                print("testing accuracy: {}".format(test_accuracy / training_step_2))
                test_acc_epoch_list.append(test_accuracy/training_step_2)
                test_loss_epoch_list.append(test_loss/training_step_2)
            # test_acc_list.append(np.max(test_acc_epoch_list))
            test_acc_list.append(test_acc_epoch_list[-1])
            test_loss_list.append(np.mean(test_loss_epoch_list))

    plot(range(0, training_epoches), test_acc_list, label='test_acc')
    plot(range(0, training_epoches), training_loss_list, label='train_loss')
    plot(range(0, training_epoches), train_acc_list, label='train_acc')
    plot(range(0, training_epoches), test_loss_list, label='test_loss')
    test_acc_list = sorted(test_acc_list, reverse=True)
    print('data_version_%d' % (data_version))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:10])))
    print("top-K mean: %.3f" % np.mean(np.array(test_acc_list[:50])))
    print(test_acc_list)
    print(mode(test_acc_list))
    title('data_version_%d' % (data_version))
    legend()

if __name__ =='__main__':

    # K.set_session(get_session(0.3, '3'))
    for i in range(5,6):
        tf.reset_default_graph()
        # transfer_learning_(i,b=10)
        transfer_learning_temp(i,b=10)

        # merge_prediction_batch(i)
        # para_prediction_batch(i)
        # prediction_(i)
        show()



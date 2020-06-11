import numpy as np
import tensorflow as tf

def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

def mask():
    # Make a 4 x 8 matrix where each row contains the length repeated 8 times.
    lengths = [4, 3, 5, 2]
    lengths_transposed = tf.expand_dims(lengths, 1)

    # Make a 4 x 8 matrix where each row contains [0, 1, ..., 7]
    range = tf.range(0, 8, 1)
    range_row = tf.expand_dims(range, 0)

    # Use the logical operations to create a mask
    mask = tf.less(range_row, lengths_transposed)

    # Use the select operation to select between 1 or 0 for each value.
    result = tf.select(mask, tf.ones([4, 8]), tf.zeros([4, 8]))


def read_csv(file_location):
    f_in =open(file_location)
    rt_list =[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(',')[0:]
        for m in range(0,len(line)):
            line[m] =float(line[m])
        rt_list.append(line)
    try:
        return np.array(rt_list,dtype=np.float32)
    except ValueError:
        return rt_list


def read_dat(file_location):
    f_in =open(file_location)
    rt_list=[]
    for i,lines in enumerate(f_in):
        line =lines.strip().split(' ')[0:]
        rt_list.append(line)
    return np.array(rt_list,dtype=np.float32)

def get_train_test_data(X_con,y,nb_x_train,nb_x_test,):
    X_train, X_test, y_train, y_test, = [], [], [], [],
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(X_con)):
            if float(nb_x_train[m]) == float(X_con[n][0]):
                X_train.append(X_con[n][1:])
                break
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(y)):
            if float(nb_x_train[m]) == float(y[n][0]):
                y_train.append(y[n][1:]-1)
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(X_con)):
            if float(nb_x_test[m]) == float(X_con[n][0]):
                X_test.append(X_con[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(y)):
            if float(nb_x_test[m]) == float(y[n][0]):
                y_test.append(y[n][1:]-1)
                break
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_train,dtype=np.int32), np.array(y_test,dtype=np.int32)

def get_train_test_data_X(input ,nb_x_train,nb_x_test):
    input_train,input_test =[],[]
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(input)):
            if float(nb_x_train[m]) == float(input[n][0]):
                input_train.append(input[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(input)):
            if float(nb_x_test[m]) == float(input[n][0]):
                input_test.append(input[n][1:])
                break
    return np.array(input_train,dtype=np.float32),np.array(input_test,dtype=np.float32)

def mse(logits, outputs):
    mse = tf.reduce_mean(tf.square(tf.subtract(logits, outputs)))
    return mse

def get_train_test_data_temp(input,nb_x_train,nb_x_test,padding_value =0,max_len =50):
    input_train, input_test = [], []
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(input)):
            if float(nb_x_train[m]) == float(input[n][0]):
                if len(input[n])>=max_len+1:
                    output =input[n][1:max_len+1]
                elif 2<=len(input[n])<max_len+1:
                    output =input[n][1:]
                    for k in range(0,max_len-len(input[n])+1):
                        output.append(padding_value)
                input_train.append(output)
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(input)):
            if float(nb_x_test[m]) == float(input[n][0]):
                if len(input[n])>=max_len+1:
                    output =input[n][1:max_len+1]
                elif 2<=len(input[n])<max_len+1:
                    output =input[n][1:]
                    for k in range(0,max_len-len(input[n])+1):
                        output.append(padding_value)
                input_test.append(output)
                break
    return np.array(input_train, dtype=np.float32), np.array(input_test, dtype=np.float32)

def get_train_test_data_Y(input ,nb_x_train,nb_x_test):
    input_train,input_test =[],[]
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(input)):
            if float(nb_x_train[m]) == float(input[n][0]):
                input_train.append(input[n][1:]-1)
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(input)):
            if float(nb_x_test[m]) == float(input[n][0]):
                input_test.append(input[n][1:]-1)
                break
    return np.array(input_train,dtype=np.int32),np.array(input_test,dtype=np.int32)

def one_hot_vector(label, size):
    output_label = []
    for i in label:
        a = np.zeros(size,)
        print(i)
        a[i] = 1
        output_label.append(a)
    return output_label

def clipped(x):
    # this handles cases when y * tf.log(y') outputs NaN
    return tf.clip_by_value(x, 1e-10, 1.0)

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    '''label_dense ndarray
    return labels_one_hot ndarray'''
    num_labels = labels_dense.shape[0]
    # print(num_labels)
    index_offset = np.arange(num_labels) * num_classes
    # print(index_offset)
    labels_one_hot = np.zeros((num_labels, num_classes))
    # print(labels_one_hot.shape)
    # assign the value to matrix
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



def sample_code():
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(y_predict, keep_prob)

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    def inference(input_tensor, weights1, biases1, weights2, biases2):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(image_batch, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


def cross_entropy_1(y,predicted_y):
    return tf.reduce_mean(-tf.reduce_sum(predicted_y * tf.log(y) + (1 - predicted_y) * tf.log(1 - y)))

def cross_entropy_2(y, predicted_y):
    return tf.reduce_mean(-tf.reduce_sum(predicted_y * tf.log(y), reduction_indices=[1]))

def squared_error(target, activation):
    return tf.reduce_sum(tf.squared_difference(target, activation),name='squared_error')


def num_linenum():
    f_in=open('number.csv')
    f_out=open('line_number.csv','w')
    for i,lines in enumerate(f_in):
        line =lines.strip().split(' ')[0]
        f_out.write('%s,%d\n'%(line,(i+1)))


def get_merge_train_test_data(X_feature,X_line_num,y,nb_x_train,nb_x_test,):
    X_train, X_test, y_train, y_test, = [], [], [], [],
    X_train_lineNum,X_test_lineNum =[],[]
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(X_feature)):
            if float(nb_x_train[m]) == float(X_feature[n][0]):
                X_train.append(X_feature[n][1:])
                break
    for m in range(0, len(nb_x_train)):
        for n in range(0, len(y)):
            if float(nb_x_train[m]) == float(y[n][0]):
                y_train.append(y[n][1:]-1)
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(X_feature)):
            if float(nb_x_test[m]) == float(X_feature[n][0]):
                X_test.append(X_feature[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(y)):
            if float(nb_x_test[m]) == float(y[n][0]):
                y_test.append(y[n][1:]-1)
                break

    for m in range(0, len(nb_x_train)):
        for n in range(0, len(X_line_num)):
            if float(nb_x_train[m]) == float(X_line_num[n][0]):
                X_train_lineNum.append(X_line_num[n][1:])
                break
    for m in range(0, len(nb_x_test)):
        for n in range(0, len(X_line_num)):
            if float(nb_x_test[m]) == float(X_line_num[n][0]):
                X_test_lineNum.append(X_line_num[n][1:])
                break
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_train,
                dtype=np.int32), np.array(y_test, dtype=np.int32), np.array(X_train_lineNum), np.array(X_test_lineNum)


def mask_output(output, seq_length):
    # masking the output vector
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int( output.get_shape()[2] )
    index = tf.range(0, batch_size) * max_length + (seq_length - 1)
    flat = tf.reshape(output, [-1, out_size])
    return tf.gather(flat, index)

def padding_and_generate_mask(x, y, new_x, new_y, new_mask_x):
    for i, (x, y) in enumerate(zip(x, y)):
        # whether to remove sentences with length larger than maxlen
        if len(x) <= max_len:
            new_x[i, 0:len(x)] = x
            new_mask_x[0:len(x), i] = 1
            new_y[i] = y
        else:
            new_x[i] = (x[0:max_len])
            new_mask_x[:, i] = 1
            new_y[i] = y
    new_set = (new_x, new_y, new_mask_x)
    del new_x, new_y
    return new_set


def masked_loss(loss, mask):
    """Softmax cross-entropy loss with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
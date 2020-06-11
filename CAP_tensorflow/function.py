import tensorflow as tf
def func():
    % matplotlib inline
    import matplotlib.pyplot as plt
    from keras import backend as K

    def get_layer_outputs():
        test_image = YOUR IMAGE GOES HERE!!!
        outputs = [layer.output for layer in model.layers]  # all layer outputs
        comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                      outputs]  # evaluation functions

        # Testing
        layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
        layer_outputs = []

        for layer_output in layer_outputs_list:
            print(layer_output[0][0].shape, end='\n-------------------\n')
            layer_outputs.append(layer_output[0][0])

        return layer_outputs

    def plot_layer_outputs(layer_number):
        layer_outputs = get_layer_outputs()

        x_max = layer_outputs[layer_number].shape[0]
        y_max = layer_outputs[layer_number].shape[1]
        n = layer_outputs[layer_number].shape[2]

        L = []
        for i in range(n):
            L.append(np.zeros((x_max, y_max)))

        for i in range(n):
            for x in range(x_max):
                for y in range(y_max):
                    L[i][x][y] = layer_outputs[layer_number][x][y][i]

        for img in L:
            plt.figure()
            plt.imshow(img, interpolation='nearest')


def return_sequence_equals_true():
    # tf Graph input
    x = tf.placeholder("float", [None, seq_max_len, 1])
    y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def dynamicRNN(x, seqlen, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, seq_max_len, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    pred = dynamicRNN(x, seqlen, weights, biases)



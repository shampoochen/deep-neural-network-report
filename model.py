import tensorflow as tf

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def inference(image_holder,batch_size):
    with tf.variable_scope('conv1') as scope:
        weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
        kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
        bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
        conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1),name=scope.name)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope('conv2') as scope:

        weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
        kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
        bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2),name=scope.name)
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')


    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3,name=scope.name)

    with tf.variable_scope('fc2') as scope:
        weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
        bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
        local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4,name=scope.name)

    with tf.variable_scope('logits') as scope:

        weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
        bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
        logits = tf.add(tf.matmul(local4, weight5), bias5,name=scope.name)
    return logits

def loss(logits, labels):
    #      """Add L2Loss to all the trainable variables.
    #      Add summary for "Loss" and "Loss/avg".
    #      Args:
    #        logits: Logits from inference().
    #        labels: Labels from distorted_inputs or inputs(). 1-D tensor
    #                of shape [batch_size]
    #      Returns:
    #        Loss tensor of type float.
    #      """
    #      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


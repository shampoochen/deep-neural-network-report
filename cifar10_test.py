import tensorflow as tf
import cifar10_input
import math,model,sys
import numpy as np

data_dir = './cifar-10-batches-bin'

batch_size = 128
num_examples = 10000

images_test, labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# logits = model.inference(image_holder,batch_size)
# top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

logits = model.inference(images_test,batch_size)
top_k_op = tf.nn.in_top_k(logits, labels_test, 1)


with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess,sys.argv[1])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    tf.train.start_queue_runners(sess=sess)
    # sess.run(tf.global_variables_initializer())



    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch,label_batch = sess.run([images_test,labels_test])
        # predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
        #                                              label_holder:label_batch})

        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1 = %.3f' % precision)

    coord.request_stop()
    coord.join(threads)
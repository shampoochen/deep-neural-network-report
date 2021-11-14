import cifar10,cifar10_input
import tensorflow as tf
import time
import model

max_steps = 3000
batch_size = 128
data_dir = './cifar-10-batches-bin'

cifar10.maybe_download_and_extract()


images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)


image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

logits = model.inference(image_holder,batch_size)


loss = model.loss(logits, label_holder)


train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #0.72

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

ckpt_path = './ckpt/cifar10-cnn-model.ckpt'
saver = tf.train.Saver()

# sess = tf.InteractiveSession()
with tf.Session() as sess:


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())
    ###
    for step in range(max_steps):
        start_time = time.time()
        image_batch,label_batch = sess.run([images_train,labels_train])
        _, loss_value = sess.run([train_op, loss],feed_dict={image_holder: image_batch,
                                                             label_holder:label_batch})
        duration = time.time() - start_time

        if (step+1) % max_steps == 0:
            save_path = saver.save(sess,ckpt_path,global_step=step)
            print("Model saved in file:%s"%save_path)

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

    save_path = saver.save(sess, ckpt_path, global_step=step)

    coord.request_stop()
    coord.join(threads)

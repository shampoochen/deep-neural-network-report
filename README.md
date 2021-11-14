# deep-neural-network-report
## Introduction to cifar-10 dataset

Cifar-10 is a small data set for pervasive objects organized by Hinton's students Alex krizhevsky and Ilya sutskever. It contains 10 categories of RGB color pictures: aircraft, cars, birds, cats, deer, dogs, frogs, horses and ships:


## Build convolution network model
__Establish a function to initialize weight, and use tf.truncated_Normal truncated normal distribution initialization weight__
```
def variable_with_weight_loss(shape, stddev, w1):     
    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))     
    if w1 is not None:        
       weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')     
       tf.add_to_collection('losses',weight_loss)     
    return var
```
__Start creating a volume stack__
Each convolutional layer in convolutional neural network is composed of several convolution units, and the parameters of each convolution unit are optimized by back propagation algorithm. The purpose of convolution operation is to extract different input features. The first convolution layer may only extract some low-level features, such as edges, lines and angles. Networks with more layers can iteratively extract more complex features from low-level features.
```
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64],stddev = 5e-2,w1=0)
kernel1 = tf.nn.conv2d(image_holder,weight1,[1, 1, 1, 1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
pool1 = tf.nn.max_pool(conv1,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding = 'SAME')
norm1 = tf.nn.lrn(pool1,4,bias = 1.0,alpha=0.001/0.9,beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64],stddev = 5e-2,w1=0)
kernel2 = tf.nn.conv2d(norm2,weight2,[1, 1, 1, 1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
norm2 = tf.nn.lrn(conv2,4,bias = 1.0,alpha=0.001/0.9,beta=0.75)
pool2 = tf.nn.max_pool(norm2,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding = 'SAME')
```

__Create full connected layer__
Fully connected layers (FC) play the role of "Classifier" in the whole convolutional neural network. If the operations of volume layer, pool layer and activation function layer map the original data to the hidden layer feature space, the full connection layer maps the learned "distributed feature representation" to the sample tag space.
```
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[-1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192,10], stddev=1/192, w1=0.004)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5) + bias5)
```

## Training network model
__Use cifar10_input to generate the data needed for training__
```
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)
```
__Create placeholder for input data, including features and labels__
```
image_holder = tf.placeholder(tf.float32,[batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32,[batch_size])
```
__Logits are obtained by the information method of neural network model module__
```
logits = model.inference(image_holder,batch_size)
loss = model.loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) 
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
```
__Start training__
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    ###
    for step in range(max_steps):
        start_time = time.time()
        image_batch,label_batch = sess.run([images_train,labels_train])
        _, loss_value = sess.run([train_op, loss],feed_dict={image_holder: image_batch,
                                                             label_holder:label_batch})
        duration = time.time() - start_time

        if step % 10 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
```
## Test model
There are 10000 samples in the test set, and the fixed batch is still used_ Size, enter test data one batch after another; Use cifar10_ The input.inputs function generates test data without flipping the picture, modifying the brightness and contrast, directly cutting the 24 x 24 block in the middle of the picture, and converting the data table: create the placeholder of the input data, including picture features and labels. Create the data size of the placeholder
```
batch_size = 128
num_examples = 10000

images_test, labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,
                                                batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

logits = model.inference(image_holder,batch_size)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,sys.argv[1])
    # 启动协程，防止陷入死锁
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    tf.train.start_queue_runners(sess=sess)

    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch,label_batch = sess.run([images_test,labels_test])
        predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
                                                     label_holder:label_batch})
        true_count += np.sum(predictions)
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1 = %.3f' % precision)

    coord.request_stop()
    coord.join(threads)
```
   

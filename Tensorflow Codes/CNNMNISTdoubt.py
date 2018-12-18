# Loading the required packages
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
# for importing the layers present in contrib
import tensorflow.contrib.layers as layers

# Now we load the mnist dataset
# each handwritten digit is of the size 28*28 i.e. 784 pixels grayscale image
from tensorflow.examples.tutorials.mnist import input_data
# we use one hot encoding for labels
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Code to visualize data
'''
i = 55
img = mnist.train.images[i]
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(mnist.train.labels[i])
'''

# getting the idea of the training set
# 55000 examples with grayscale image of 784 pixels
print(mnist.train.images.shape)
# the labels are one hot encoded
print(mnist.train.labels.shape)
# 10000 test examples
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# you can also not specify the shape None means it can take any number of examples
x = tf.placeholder(tf.float32 ,shape = [None , 784] , name='X')
y = tf.placeholder(tf.float32,shape = [None , 10],name='Y')
keep_prob = tf.placeholder(tf.float32)

# Parameters
learning_rate = 0.001
training_iters = 50
batch_size = 128
display_step = 10
no_hidden_units = 1000

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.85 # Dropout, probability to keep units

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    # reshape the input picture
    # -1 means just take care of the total number of inputs it may be 1 or 55000
    # because we are not sure of the total size of the input data test or train
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # First convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Output size is 28*28*32
    # Max Pooling used for downsampling
    # output is 14*14*32
    conv1 = maxpool2d(conv1, k=2)

    # Second convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Output is (14,14,64)
    # Max Pooling used for downsampling
    conv2 = maxpool2d(conv2, k=2)
    # Output is (7,7,64)

    # Reshape conv2 output to match the input of fully connected layer 
    # To get the shape as a list of ints, do tensor.get_shape().as_list()
    # the shape is -1,7*7*64
    shaping = tf.reshape(conv2, [-1,7*7*64])

    fc1 = layers.fully_connected(shaping, no_hidden_units, activation_fn=tf.nn.relu )
    fc2 = layers.fully_connected(fc1, no_hidden_units, activation_fn=tf.nn.relu)
    # We are applying softmax activation in the output layer 
    out = layers.fully_connected(fc2, n_classes, activation_fn=None)
    return out

weights = {
    # 5x5 conv, 1 input, and 32 outputs( or 32 filters)
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, and 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
}

pred = conv_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

train_loss = []
train_acc = []
test_acc = []

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            loss_train, acc_train = sess.run([cost, accuracy], 
                                             feed_dict={x: batch_x,
                                                        y: batch_y,
                                                        keep_prob: 1.})
            print ("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.2f}".format(loss_train) + ", Training Accuracy= " + \
                  "{:.2f}".format(acc_train))
    
            # Calculate accuracy for mnist test images. 
            # Note that in this case no dropout
            acc_test = sess.run(accuracy, 
                                feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      keep_prob: 1.})
    
            print ("Testing Accuracy:" + \
               "{:.2f}".format(acc_train))
    
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)
            
        step += 1

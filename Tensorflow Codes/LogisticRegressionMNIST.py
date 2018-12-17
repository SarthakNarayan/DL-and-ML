# NOT SURE WITH THE USE OF SCOPE AND TENSORBOARD PROPERLY DOCUMENTATION WILL BE UPDATED FURTHER REGARDING THIS
# Loading the required packages
import tensorflow as tf
import matplotlib.pyplot as plt , matplotlib.image as mpimg
import numpy as np

# Now we load the mnist dataset
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

# Assigning weights and biases to zeroes 
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# also try with random initialization of weights and biases
# W = tf.Variable(tf.random_normal([784, 10] , seed = 12), name='W')
# b = tf.Variable(tf.random_normal([10] , seed = 12), name='b')

# Define the model
# Here softmax is the activation function
with tf.name_scope("wx_b") as scope:
    y_hat = tf.nn.softmax(tf.matmul(x,W) + b)
    
#Add summary ops to collect data while training
# we can see histogram summary of how weights and biases change relative to time
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# Define the cross-entropy loss function
# We use the scalar summary to obtain the variation of loss function over time
# Here cross-entropy is the loss function
with tf.name_scope('cross-entropy') as scope:
    # logits is the prediction , labels are the true values
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    tf.summary.scalar('cross-entropy', loss)
    
# Choose the optimizer for training
with tf.name_scope('Train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Merge All summaries
merged_summary_op = tf.summary.merge_all()

'''
Use of tf.equal and argmax
import tensorflow as tf

a = [[1,9,3,4],[1,5,6,2]]
b = [1,9,3,4]
with tf.Session() as sess:
    print(sess.run(tf.argmax(b)))
    # along the column 
    # always remember in 2d array 0 is along the column and 1 is along the row
    print(sess.run(tf.argmax(a,0)))
    print(sess.run(tf.argmax(a,1)))
    print(sess.run(tf.equal(a,b)))
    print(sess.run(tf.cast(tf.equal(a,b) , tf.float32)))
'''
'''
Use of reduce_mean
import tensorflow as tf

a = tf.constant([[100, 110], [10, 20], [1000, 1100]])

# Use reduce_mean to compute the average (mean) across a dimension.
b = tf.reduce_mean(a)
c = tf.reduce_mean(a, axis=0)
d = tf.reduce_mean(a, axis=1)
session = tf.Session()

print("INPUT")
print(session.run(a))
print("REDUCE MEAN")
print(session.run(b))
print("REDUCE MEAN AXIS 0")
print(session.run(c))
print("REDUCE MEAN AXIS 1")
print(session.run(d))

OUTPUT:-
INPUT
[[ 100  110]
 [  10   20]
 [1000 1100]]
REDUCE MEAN
390
REDUCE MEAN AXIS 0
[370 410]
REDUCE MEAN AXIS 1
[ 105   15 1050]
'''
# This piece of segment is only used for predicting on a number of samples together
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
# Converting true or false to 1.0 or 0.0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

max_epochs = 20
batch_size = 100
with tf.Session() as sess:
    # initialize all variables
    sess.run(init)  
    # Create an event file for visualization in tensorboard
    summary_writer = tf.summary.FileWriter('graphs', sess.graph)  
    
    # Training
    for epoch in range(max_epochs):
        loss_avg = 0
        # calculating the number of batches in our case 5500
        num_of_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_of_batch):
            # get the next batch of data 
            batch_xs, batch_ys = mnist.train.next_batch(100)  
            # from feed dict we give 100 training examples at a time
            _, l, summary_str = sess.run([optimizer,loss, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})  # Run the optimizer
            # we add the loss of whole training set since we get the loss in batches
            loss_avg += l
            summary_writer.add_summary(summary_str, epoch*num_of_batch + i)  # Add all summaries per batch
        
        # after getting the total loss of the whole training set we divide it by the number of batches
        loss_avg = loss_avg/num_of_batch
        print('Epoch {0}: Loss {1}'.format(epoch, loss_avg))
    
    print('Done')
    
    # for checking the accuracy of a lot of examples
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))
    
    # Storing the weights and biases for individual prediction
    weights = sess.run(W)
    biases = sess.run(b)
    
# input shape is (784,) but the predictor takes (none , 784) so it needs to be reshaped
# this was not needed when we were predicting on a number of examples because the shape was (10000 , 784)
print(mnist.test.images[1].shape)

def Predictor(X):
    reshaped_input = np.reshape(X , (1 , 784))
    print(reshaped_input.shape)
    # predicting the output
    y_prediction = tf.nn.softmax(tf.matmul(reshaped_input,weights) + biases)
    
    # do not do np.argmax on tensorflow variables or any other function because outside
    # a session these are only nodes in a graph hence u will never get the correct result
    #print(np.argmax(y_prediction[0]))   wrong
    # but u can do normal numpy other operations on variables which are not part of tensorflow graph like below operation
    
    print(np.argmax(mnist.test.labels[1]))
    with tf.Session() as sess:
        # since the matrix is like [[]] so if you dont give [0] it will always give 0
        print(sess.run(tf.argmax(y_prediction[0])))
        # it is a 1*10 elements with the probability of each digit using argmax we get the position of the 
        # digit with max probability
        print(sess.run(y_prediction[0]))

# Comment the predictor incase ur predicting over the entire test set    
Predictor(mnist.test.images[1])

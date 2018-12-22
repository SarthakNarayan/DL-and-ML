import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784] , name = 'x')

# The code looks very similar to normal MLP but with no activation func i.e. linear
# By default activation is none
def modeltype(choice = 1):
    if (choice == 1):
        fc1 = layers.fully_connected(X, 300)
        fc2 = layers.fully_connected(fc1, 75)
        fc3 = layers.fully_connected(fc2, 300)
        out = layers.fully_connected(fc3, 784)
        return out
    
    if (choice == 2):
        '''
        shape of X is only partially defined during the construction
        phase, we cannot know in advance the shape of the noise that
        we must add to X. We cannot call X.get_shape() because this
        would just return the partially defined shape of X ([None,
        n_inputs]), and random_normal() expects a fully defined shape so
        it would raise an exception. Instead, we call tf.shape(X)
        '''
        X_noisy = X + tf.random_normal(tf.shape(X))
        fc1 = layers.fully_connected(X_noisy, 300)
        fc2 = layers.fully_connected(fc1, 75)
        fc3 = layers.fully_connected(fc2, 300)
        out = layers.fully_connected(fc3, 784)
        return out
    
        
output = modeltype(2)
# MSE error
reconstruction_loss = tf.reduce_mean(tf.square(output - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(0.01)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 150

n_test_digits = 2
X_test = mnist.test.images[:n_test_digits]

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    plt.show()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        num_of_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_of_batch):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # We dont feed the labels as it is unsupervised learning
            sess.run(training_op, feed_dict={X: X_batch })
    outputs_val = output.eval(feed_dict={X: X_test})

plot_image(outputs_val[0])
plot_image(X_test[0])
plot_image(outputs_val[1])
plot_image(X_test[1])
    

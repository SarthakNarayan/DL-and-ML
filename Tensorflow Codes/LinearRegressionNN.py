import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Its better to load data from sklearn than from tensorflow
boston_dataset = datasets.load_boston()
print(boston_dataset.data.shape)
print(boston_dataset.target.shape)

originial_X_train = boston_dataset.data
original_Y_train = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(originial_X_train , 
                                                    original_Y_train , 
                                                    test_size=0.3, 
                                                    random_state=0)

print(" Training data size")
print(X_train.shape , y_train.shape)
print(" Testing data size")
print(X_test.shape , y_test.shape)

# Data has to be rescaled as fit_transform only takes 2d array
y_train_reshape = np.reshape(y_train , (354,1))
y_test_reshape = np.reshape(y_test , (152,1))

# Normalize the inputs
# By scaling all the values between 0 and 1
X_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train)
y_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(y_train_reshape)

# Output will also have to be scaled because we will be sigmoid activation function
# If we use tanh activation function then its better to use feature_range=(-1, 1)
# when we did linear regression there was no activation function hence only the inputs were scaled
X_test =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X_test)
y_test =  MinMaxScaler(feature_range=(0, 1)).fit_transform(y_test_reshape)

print(" Labels shape after scaling")
print(y_test.shape , y_train.shape)

n_hidden = 500
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu)
    fc2 = layers.fully_connected(fc1, n_hidden, activation_fn=tf.nn.relu)
    fc3 = layers.fully_connected(fc2, n_hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc3, 1, activation_fn=tf.tanh)
    return out


# build model, loss, and train op
# in neural networks it is important to specify the input shape
x = tf.placeholder(tf.float32, name='X' , shape = [None , 13])
y = tf.placeholder(tf.float32, name='Y')
y_hat = multilayer_perceptron(x)

loss =  tf.reduce_mean(tf.cast(tf.square(y -y_hat), tf.float32))
train = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

init = tf.global_variables_initializer()

test_example_number = 100
max_epoch = 500
def Prediction(X):
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # train the model for 100 epcohs
        for i in range(max_epoch):
           _, l, p = sess.run([train, loss, y_hat], feed_dict={x: X_train, y: y_train})
           print('Epoch {0}: Loss {1}'.format(i, l))
    
        print("Training Done")
        print("Optimization Finished!")
    
        # Calculate accuracy
        #print(" Mean Squared Error (Test data):", loss.eval({x: X_test, y: y_test}))
        
        print(X.shape)
        reshaped_input = np.reshape(X , (1 , 13))
        y_prediction = y_hat.eval(feed_dict = {x : reshaped_input})
        print(" The original value is : " , end = " ")
        print(y_test[test_example_number][0])
        print(" The predicted value is : " , end = " ")
        print(y_prediction[0][0])

Prediction(X_test[test_example_number])

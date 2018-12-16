# Loading the required packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize the input data(array form)
# Very important
def Normalize(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)
    X = (x - mean)/standard_deviation
    return X

# Now we load the boston house price dataset using tensorflow contrib Datasets
# Separate the data into X_train and Y_train
boston_dataset = tf.contrib.learn.datasets.load_dataset('boston')
print(boston_dataset.data.shape)
# we are considering all the features regression
X_train , Y_train = boston_dataset.data , boston_dataset.target
print(X_train.shape , Y_train.shape)
# So we see we have 506 samples of training data with 13 features
X_train = Normalize(X_train)
n_samples = X_train.shape[0]
n_features = X_train.shape[1]

# Placeholder for storing the training data
# even if we donot specify the input shape it doesnot matter and tensorflow will adjust accordingly
X = tf.placeholder(tf.float32 , name = 'X' , shape = [n_samples , n_features])
Y = tf.placeholder(tf.float32 , name = 'Y')

# Assigning weights randomly in a 13*1 matrix
w = tf.Variable(tf.random_normal([n_features , 1] , seed = 12) , name = 'weight' )
# bias is a 506*1 matrix with seed for constant results
b = tf.Variable(tf.random_normal([n_samples , 1],  seed = 12) , name = 'biases' )

# Linear Regression Model for prediction
# matmul is important since we are multiplying matrices we cannot use * gives absurd results
y_hat = tf.add(tf.matmul(X,w) , b) 

# Loss function i.e. mean squared error with l2 regularization and 0.6 is the lambda
# tf.reduce_mean is important otherwise it will give errors i.e. you will get very high loss nan
loss = tf.reduce_mean(tf.square( Y - y_hat, name = 'loss')) + 0.01*tf.nn.l2_loss(w)
#loss = tf.square( Y - y_hat, name = 'loss') + 0.6*tf.nn.l2_loss(w)  will give errors

# Gradient descent optimiter for minimizing loss
# if your learning rate is too high it will not converge and will give nan as output
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# Initializing variables
init = tf.global_variables_initializer()
total = []

no_of_epochs = 100
with tf.Session() as sess:
    # initializing the variables
    sess.run(init)
    # for visualization in tensorboard
    writer = tf.summary.FileWriter('graphs' , sess.graph)
    for i in range(no_of_epochs):
        _ , l = sess.run([optimizer , loss] , feed_dict = {X : X_train , Y : Y_train})
        # no need of zip as we are multiplying the whole X and W matrix
        # in simple regression since w was 1*1 matrix we had to extract each term then multiply it
        total.append(l)
        print('Epoch {0}: Loss {1}'.format(i, l))

    writer.close()
    # getting the value of weight and bias after 100 epochs
    w_value , b_value  = sess.run([w , b ])

# * wont work matmul is important
Y_pred = np.matmul(X_train,w_value) + b_value
# you can make your own X_train
print('done')
# we aee Y_pred is of shape (506,1)
print(Y_pred.shape)

# You can compare Y_pred and Y_train 
plt.plot(total)
plt.show()

# if you want to predict values
def Predictor(X):
    Y_prediction = np.matmul(X,w_value) + b_value
    # any number in the array will do because we are getting a 506*1 array due to broadcating of X from 1*13 to 506*13
    # so any value u take in the array it will be same
    print(Y_prediction[0])

# now the input to Predictor should be a matrix of 1*13

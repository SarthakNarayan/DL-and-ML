# Loading the required packages
# hi what is your name
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize the input data(array form)
def Normalize(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)
    X = (x - mean)/standard_deviation
    return X

'''
Here we are building the computational graph
'''
# Now we load the boston house price dataset using tensorflow contrib Datasets
# Separate the data into X_train and Y_train
boston_dataset = tf.contrib.learn.datasets.load_dataset('boston')
print(boston_dataset.data.shape)
# By 5 means out of 13 columns(features) we are only taking one into consideration
X_train , Y_train = boston_dataset.data[:,5] , boston_dataset.target
print(X_train.shape , Y_train.shape)
# So we see we have 506 samples of training data
X_train = Normalize(X_train)
n_samples = len(X_train)

# Placeholder for storing the training data
X = tf.placeholder(tf.float32 , name = 'X')
Y = tf.placeholder(tf.float32 , name = 'Y')

# Assigning weights and biases to 0
w = tf.Variable(0.0 , name = 'weight')
b = tf.Variable(0.0 , name = 'biases')

# Linear Regression Model for prediction
y_hat = X*w + b

# Loss function
loss = tf.square( Y - y_hat, name = 'loss')

# Gradient descent optimiter for minimizing loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Initializing variables
init = tf.global_variables_initializer()
total = []

'''
Done building the computational graph
'''

'''
Use of zip

# initializing list of players. 
players = [ "Sachin", "Sehwag", "Gambhir", "Dravid", "Raina" ] 
# initializing their scores 
scores = [100, 15, 17, 28, 43 ] 
# printing players and scores. 
for pl, sc in zip(players, scores): 
    print ("Player :  %s     Score : %d" %(pl, sc)) 
Output:

Player :  Sachin     Score : 100
Player :  Sehwag     Score : 15
Player :  Gambhir     Score : 17
Player :  Dravid     Score : 28
Player :  Raina     Score : 43
'''

no_of_epochs = 100
with tf.Session() as sess:
    # initializing the variables
    sess.run(init)
    # for visualization in tensorboard
    writer = tf.summary.FileWriter('graphs' , sess.graph)
    for i in range(no_of_epochs):
        total_loss = 0
        for x,y in zip(X_train , Y_train):
            # _ because we dont want to store optimizers value
            _ , l = sess.run([optimizer , loss] , feed_dict = {X : x , Y : y})
            # we are calculating total loss for all the sample of data going through them one by one
            # In multiple regression we multiply the data all at once in a matrix
            total_loss += l
        # we divide by n_samples we are going through each sample in above for loop and then adding the loss
        total.append(total_loss/n_samples)
        print('Epoch {0} : Loss {1}'.format(i , total_loss/n_samples))
    writer.close()
    # getting the value of weight and bias after 100 epochs
    w_value , b_value = sess.run([w , b])

Y_pred = X_train*w_value + b_value
print('done')

# Plotting the result
plt.plot(X_train, Y_train, 'bo', label='Real Data')
plt.plot(X_train,Y_pred,  'r', label='Predicted Data')
plt.legend()
plt.show()

plt.plot(total)
plt.show()

# Loss gets stuck at one value due to normalization comment it and see the results
# Also see multiple regression example for clearer concept

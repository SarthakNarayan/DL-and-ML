# We see why neural networks are no good for images and we tend to go for CNN


import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense , Dropout
import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10

# loading the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
# to diplay the samples
plt.imshow(x_train[5] , cmap = plt.cm.binary)
plt.show()
'''
# get an idea of shape 
print("Getting an idea of shape")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# data before normalizing
#print(x_train[5])

# normalizing the data i.e. making the pixel values between 0 and 1 as compared to 255 and 0
# data has to be normalized for both training and test set
# or x_train = x_train/255 same for test
x_train = normalize(x_train , axis = 1)
x_test = normalize(x_test , axis = 1)

# data after normalizing
#print(x_train[5])

n_hidden = 300
# Our image is (None,28,28,1) so before feeding it to the neural network we need to flatten it
# i.e. (None , 28*28*1)
# -1 means I dont care about the number of samples
x_train = np.reshape(x_train , (-1,3072))
x_test = np.reshape(x_test , (-1,3072))

print("Data shape after reshaping" )
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
# if we see the shape of labels it is not one hot encoded which has to be done
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)

model = Sequential()
model.add(Dense(n_hidden, activation='relu' , input_dim=(3072) , name = 'layer1'))
model.add(Dropout(0.2))
model.add(Dense(n_hidden, activation='relu' , name = 'layer2'))
model.add(Dropout(0.2))
model.add(Dense(n_hidden, activation='relu' , name = 'layer3'))
model.add(Dropout(0.2))
model.add(Dense(n_hidden, activation='relu' , name = 'layer4'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax' , name = 'output'))
#model.summary()

# metrics refers to the thing we want to calculate
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# Training the model
model.fit(x_train , y_train , epochs=5  , validation_split=0.2)

test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , (test_accuracy)*100)

# prediction
# For prediction u need to alter the shape a bit because x_test[].shape for a single example is (784,)
# It has to be of shape (1,784)
#print(x_test[0].shape)

test_no = 5
def Predictor(X):
    X = np.reshape(X , (1,3072))
    prediction = model.predict(X)
    print(np.argmax(prediction[0]))
    print(" To validate it")
    print(y_test[test_no])
    
Predictor(x_test[test_no])

# With a very bad CNN model we obtained 70% accuracy in just 5 epochs
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense , Dropout , Conv2D , MaxPooling2D , Flatten
import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
# we use different time stamp for naming because while training we may forget to change the name
# random can also be used but there is a chance random may give us same values
import time

NAME = "CNN.{}".format(int(time.time()))
# for callbacks
# logdir is the location where the tensorboard will be present
tensorboard = TensorBoard(log_dir= 'graphs/{}'.format(NAME))

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

# if we see the shape of labels it is not one hot encoded which has to be done
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)

model = Sequential()
model.add(Conv2D( 64 , (5,5) , activation='relu' , input_shape = (32,32,3) , padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D( 128 , (5,5) , activation='relu' , padding='same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(100 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10 , activation='softmax'))
#model.summary()

# metrics refers to the thing we want to calculate
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# Training the model
model.fit(x_train , y_train , epochs=5  , validation_split=0.2 , batch_size=200 , callbacks=[tensorboard])

test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , (test_accuracy)*100)

# prediction
# For prediction u need to alter the shape a bit because x_test[].shape for a single example is (784,)
# It has to be of shape (1,784)
#print(x_test[0].shape)

'''
test_no = 5
def Predictor(X):
    X = np.reshape(X , (1,32,32,3))
    prediction = model.predict(X)
    print(np.argmax(prediction[0]))
    print(" To validate it")
    print(y_test[test_no])
    
Predictor(x_test[test_no])
'''
# just go the graph folder in anaconda prompt and type
# tensorboard --logdir=./

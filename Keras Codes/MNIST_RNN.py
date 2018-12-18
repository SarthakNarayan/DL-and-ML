# With a very bad RNN model we obtained 97% accuracy in just 5 epochs which is simply amazing

from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense , Dropout , CuDNNLSTM 
import numpy as np
from keras.datasets import mnist
from keras.optimizers import adam
from keras.utils import np_utils

# loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
# VERY IMPORTANT SOMETIMES THERE ARE CHANCES WHEN THERE MAY NOT BE ANY LEARNING WITHOUT NORMALIZATION
x_train = normalize(x_train , axis = 1)
x_test = normalize(x_test , axis = 1)

# data after normalizing
#print(x_train[5])

n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)

model = Sequential()
# input to LSTM cannot be only 784
model.add(CuDNNLSTM(128 , input_shape = (28,28) , return_sequences= True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10 , activation='softmax'))
#model.summary()

optimizer = adam(decay = 0.00005)
# metrics refers to the thing we want to calculate
model.compile(optimizer = optimizer , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# Training the model
model.fit(x_train , y_train , epochs=5  , validation_split=0.2 , batch_size=200)

test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , (test_accuracy)*100)

# prediction
# For prediction u need to alter the shape a bit because x_test[].shape for a single example is (28*28,)
# It has to be of shape (1,28,28)
#print(x_test[0].shape)

test_no = 5
def Predictor(X):
    X = np.reshape(X , (1,28,28))
    prediction = model.predict(X)
    print(np.argmax(prediction[0]))
    print(" To validate it")
    print(y_test[test_no])
    
Predictor(x_test[test_no])


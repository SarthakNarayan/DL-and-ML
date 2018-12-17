from sklearn import datasets
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

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
# Build Model
model = Sequential()
# Only the number of input features need to be specified for the first layer
model.add(Dense(n_hidden, input_dim=(13), activation='relu' , name = 'layer1'))
model.add(Dense(n_hidden, activation='relu' , name = 'layer2'))
model.add(Dense(n_hidden, activation='relu' , name = 'layer3'))
model.add(Dense(1, activation='sigmoid', name = 'output'))
model.summary()   # Summarize the model

#Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

#Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=100, batch_size=50, verbose=1)

# is expecting a 2D array of shape (num_instances, features), like X_test is.
# But indexing a single instance as in X_test[10] returns a 1D array of shape (features,)
predicting_sample = 100
reshaping_X = np.reshape(X_test[predicting_sample] , (1,13))
prediction = model.predict(reshaping_X)
print("The prediction is :" , prediction[0][0])
print(" The original value is :" , y_test[0][0])

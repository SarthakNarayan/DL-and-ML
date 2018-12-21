# AMAZING RESULTS

from sklearn.svm import LinearSVR
from sklearn.datasets import load_boston
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

boston_dataset = load_boston()
#print(boston.data)
#print(boston.data.shape)

originial_X_train = boston_dataset.data
original_Y_train = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(originial_X_train , 
                                                    original_Y_train , 
                                                    test_size=0.3, 
                                                    random_state=0)

# if you dont scale the data your predictions will be absurd
X_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train)
X_test =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X_test)

svm_reg = LinearSVR(epsilon=0.001 , C=100)
svm_reg.fit(X_train, y_train)

predictions = []
for i in X_test:
    i = np.reshape(i , (1,13))
    predict = svm_reg.predict(i)
    predictions.append(predict)

predictions = np.array(predictions)

# you can also give your own 13 features to predict the value
def Tester(test):
    print("test data" , y_test[test])
    print("Predicted data" , predictions[test][0])

Tester(40)

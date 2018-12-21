# AMAZING RESULTS

from sklearn.svm import LinearSVR
from sklearn.datasets import load_boston
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

regressor_type = 4
def TypeOfRegressor(choice = 1):
    if(choice == 1):
        svm_reg = LinearSVR(epsilon=0.001 , C=100)
        return svm_reg
    
    if(choice == 2):
        rf_reg = RandomForestRegressor(n_estimators=1000 , min_samples_leaf=10 , n_jobs=1)
        return rf_reg
    
    if(choice == 3):
        dt_reg = DecisionTreeRegressor(max_depth=20 , min_samples_leaf=20)
        return dt_reg
    
    if(choice == 4):
        gbr_reg = GradientBoostingRegressor(max_depth=20, n_estimators=500, learning_rate=0.05)
        return gbr_reg

# takes time to execute
regressor = TypeOfRegressor(regressor_type)
regressor.fit(X_train, y_train)

predictions = []
for i in X_test:
    i = np.reshape(i , (1,13))
    predict = regressor.predict(i)
    predictions.append(predict)

predictions = np.array(predictions)

# you can also give your own 13 features to predict the value
def Tester(test):
    print("test data" , y_test[test])
    print("Predicted data" , predictions[test][0])

scorer = regressor.score(X_test , y_test)
Tester(30)
if (regressor_type == 1):
    print("Linear regressor accuracy" , scorer)
if (regressor_type == 2):
    print("Random forest regressor accuracy" , scorer)
if (regressor_type == 3):
    print("Decision tree regressor accuracy" , scorer)
if (regressor_type == 4):
    print("Gradient boosting regressor accuracy" , scorer)

# ALTHOUGH THE ACCURACY OF Linear regressor IS LESS THAN Random forest regressor 
# BUT PREDICTION SCORE IS VERY CLOSE TO REAL VALUE

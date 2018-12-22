from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from keras.datasets import mnist

'''
INFO ABOUT THE DATA
targets are setosa , versicolor , virginica
50 samples in each of 3 classes
features are sepal length , sepal width , petal length ,petal width in cm\n
'''
choice_of_dataset = 3
def datasetUsed(choice = 1):
    if(choice == 1):
        iris = datasets.load_iris()
        x = iris["data"] 
        _y = iris["target"]
        return X,y
    
    if(choice == 2):
        digits = load_digits()
        x = digits["data"] 
        _y = digits["target"]
        return x,_y
    
    if(choice == 3):
        (x, _y), (x_test, y_test) = mnist.load_data()
        '''
        scikit-learn expects 2d num arrays for the training dataset for a fit
        function. The dataset you are passing in is a 3d array you need to 
        reshape the array into a 2d.
        '''
        # since there are too many training examples and takes too much time to process
        x = x[:5000]
        _y = _y[:5000]
        x = np.reshape(x , (-1,28*28))
        return x,_y 

X , y = datasetUsed(choice_of_dataset)

Yes_PCA = True
if(Yes_PCA == True):
    print("shape before compression" , X.shape)
    '''
    In n_components you can specify number of dimensions or the variance between 0 and 1
    that you want to preserve 
    '''
    pca = PCA(n_components = 0.95)
    X2D = pca.fit_transform(X)
    print("shape after compression" , X2D.shape)
else:
    print("shape without compression" , X.shape)
    X2D = X

X_train, X_test, y_train, y_test = train_test_split(X2D , 
                                                    y , 
                                                    test_size=0.3, 
                                                    random_state=0)

type_classifier = 1

def choiceofclassifier(choice = 1):
    if(choice == 1):   
        return LogisticRegression()

    if(choice==2):
        return SVC() 
    
    if(choice == 3):
        '''
        The following code trains a Random Forest classifier with 500 trees 
        (each limited to maximum 16 nodes), using all available CPU cores:
        '''
        return RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=1)
    
    if(choice == 4):
        return KNeighborsClassifier()
    
    if(choice == 5):
        return DecisionTreeClassifier()
    
    if(choice == 6):
        '''
        The following code trains an ensemble of 500 Decision Tree classifiers,5 each trained on 
        100 training instances randomly sampled from the training set with replacement 
        (this is an example of bagging, but if you want to use pasting instead, just set bootstrap=False). 
        '''
        bag_clf = BaggingClassifier(
                DecisionTreeClassifier(), n_estimators=500,
                max_samples=100, bootstrap=True, n_jobs=1)
        return bag_clf
    
    if(choice == 7):
        ada_clf = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=10), n_estimators=200,
                learning_rate=0.5)
        return ada_clf
    
    if(choice == 8):
        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        voting_clf = VotingClassifier(
                estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf) , ('knn' ,knn_clf)],
                voting='hard')
        return voting_clf
        
classifier = choiceofclassifier(type_classifier)
classifier.fit(X_train, y_train)


def accuracyTesting(testing_sample = 10):
    predictions = []
    for i in X_test:
        i = np.reshape(i , (-1,X_test.shape[1]))
        predict = classifier.predict(i)
        predictions.append(predict)
        
    accuracy = accuracy_score(y_test, predictions)
    predictions = np.array(predictions)
    print(predictions[testing_sample][0])
    print(y_test[testing_sample])
    return accuracy

testing_sample = 10
score = accuracyTesting(testing_sample)
if(type_classifier == 1):
    print("Logisric Regression accuracy",score)
if(type_classifier == 2):
    print("SVM accuracy",score)
if(type_classifier == 3):
    print("Random Forest accuracy",score)
if(type_classifier == 4):
    print("knn accuracy",score)
if(type_classifier == 5):
    print("decision tree accuracy",score)
if(type_classifier == 6):
    print("bagging accuracy",score)
if(type_classifier == 7):
    print("Boosting accuracy",score)
if(type_classifier == 8):
    print("Ensemble accuracy",score)


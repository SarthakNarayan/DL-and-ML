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

'''
INFO ABOUT THE DATA
targets are setosa , versicolor , virginica
50 samples in each of 3 classes
features are sepal length , sepal width , petal length ,petal width in cm\n
'''
iris = datasets.load_iris()

# For all the data
X = iris["data"] 
#print(X.shape)
y = iris["target"]

type_classifier = 7

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
        return RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    
    if(choice == 4):
        return KNeighborsClassifier()
    
    if(choice == 5):
        return DecisionTreeClassifier()
    
    if(choice == 6):
        '''
        The following code trains an ensemble of 500 Decision Tree classifiers,5 each trained on 
        100 training instances randomly sampled from the training set with replacement 
        (this is an example of bagging, but if you want to use pasting instead, just set bootstrap=False). 
        The n_jobs parameter tells Scikit-Learn the number of CPU cores to use for training
        and predictions (–1 tells Scikit-Learn to use all available cores):
        '''
        bag_clf = BaggingClassifier(
                DecisionTreeClassifier(), n_estimators=500,
                max_samples=100, bootstrap=True, n_jobs=-1)
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

def accuracyTesting(testing_sample = 10):
    X = iris["data"]
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(X , 
                                                    y , 
                                                    test_size=0.3, 
                                                    random_state=0)
    classifier.fit(X_train, y_train)
    
    predictions = []
    for i in X_test:
        i = np.reshape(i , (-1,4))
        predict = classifier.predict(i)
        predictions.append(predict)
        
    accuracy = accuracy_score(y_test, predictions)
    predictions = np.array(predictions)
    print(predictions[testing_sample][0])
    print(y_test[testing_sample])
    return accuracy

testing_sample = 1
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


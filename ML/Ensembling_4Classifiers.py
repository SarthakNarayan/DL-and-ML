from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

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

type_classifier = 4

def choiceofclassifier(choice = 1):
    if(choice == 1):   
        return LogisticRegression()

    if(choice==2):
        return SVC() 
    
    if(choice == 3):
        return RandomForestClassifier()
    
    if(choice == 4):
        return KNeighborsClassifier()
    
    if(choice == 5):
        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        voting_clf = VotingClassifier(
                estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf) , ('knn' ,knn_clf)],
                voting='hard')
        return voting_clf
        
classifier = choiceofclassifier(type_classifier)
classifier.fit(X, y)

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
    print("Ensemble accuracy",score)

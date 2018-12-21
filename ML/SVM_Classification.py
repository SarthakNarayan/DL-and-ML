from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

'''
INFO ABOUT THE DATA
targets are setosa , versicolor , virginica
50 samples in each of 3 classes
features are sepal length , sepal width , petal length ,petal width in cm\n
'''
iris = datasets.load_iris()
# to get an idea about data set
#print(iris)
#print(iris["data"])
# 150 samples with 4 features
#print(iris["data"].shape)
#print(iris["target"])

# For all the data
X = iris["data"] 
#print(X.shape)
y = iris["target"]

# if you want any particular features then prediction array has to be changed
#X = iris["data"][:,:2]

# if u want to find with only iris virginica as label
#y = (iris["target"] == 2)
# y is an array in terms of true and false
#print(y)
#y = y.astype(np.int32) 
#print(y)
#print(y.shape)

classifier_you_want = 4
def choiceofclassifier(choice = 1):
    if(choice == 1):      
        linear_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))))
        return linear_svm_clf
    
    if(choice==2):
        polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))))
        return polynomial_svm_clf
    
    if(choice == 3):
        poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))))
        return poly_kernel_svm_clf
    
    if(choice == 4):
        rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))))
        return rbf_kernel_svm_clf

classifier = choiceofclassifier(classifier_you_want)
classifier.fit(X, y)

predict = classifier.predict([[4 , 6.8 , 1.7 , 5]])
print(predict)

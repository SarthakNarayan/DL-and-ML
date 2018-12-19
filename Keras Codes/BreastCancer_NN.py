# The dataset is present in the dataset folder
# Change the data directory before proceeding

from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense , Dropout 
import numpy as np
from keras.utils import np_utils
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Loading a Pandas data
data_dir = r"C:\Users\Lenovo\Desktop\training set"
csv_name = r"data.csv"
file_name = os.path.join(data_dir , csv_name)
df = pd.read_csv(file_name)

# removing rows with 0
def RowRemover(x = 0):
    index = []
    rowiterator = []
    number_of_columns = len(df.columns)
    for i in range(number_of_columns):
        index.append(i)
        
    for i , row in df.iterrows():
        rowiterator.append(i)

    print("before dropping the rows with {}".format(x))
    print(df.shape)
    rows = []
    for i in rowiterator: 
        for j in index:
            if(df.iloc[i,j] == x):
                #print("row" , i , "column" , j)
                rows.append(i)
                break
    df.drop(rows , inplace = True)
    print("before dropping the rows with {}".format(x))
    print(df.shape)

RowRemover()

def LabelMaker():
    labels = []
    for i in df.iloc[:,1]:
        if(i == 'M'):
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels)
    print(labels.shape)
    
    n_classes = 2
    print("Shape before one-hot encoding: ", labels.shape)
    labels = np_utils.to_categorical(labels, n_classes)
    print("Shape after one-hot encoding: ", labels.shape)
    return labels

Y = LabelMaker()

# removing unwanted columns
df.drop(columns=['id', 'Unnamed: 32' , 'diagnosis'] , inplace = True)
print("after dropping unwanted columns")
print(df.shape)


features = np.array(df)
print("after converting to array")
print(features.shape)
print(Y.shape)

# shuffling the data
X_train , Y_train = shuffle(features , Y , random_state = 12)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X_train , Y_train ,
                                                        test_size = 0.2, 
                                                        random_state = 12)

print("after split")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Preprocessing the data
x_train = normalize(x_train , axis = 1)
x_test = normalize(x_test , axis = 1)

hidden_units = 200
model = Sequential()
model.add(Dense(hidden_units , input_dim = (30) , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_units , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_units , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2 , activation='softmax'))

# metrics refers to the thing we want to calculate
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.fit(x_train , y_train , epochs=100  , validation_split=0.2)

test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , (test_accuracy)*100)

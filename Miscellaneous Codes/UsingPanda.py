import pandas as pd
import os
from keras.utils import np_utils
import numpy as np

# Loading a Pandas data
data_dir = r"C:\Users\Lenovo\Desktop\training set"
csv_name = r"data.csv"
file_name = os.path.join(data_dir , csv_name)
df = pd.read_csv(file_name)
dfi = pd.read_csv(file_name)

# ----------------------------------------------------------------------------
# To view any column 
#print(df['Unnamed: 32'])
# More than one column can also be extracted in any order
#print(df[['Unnamed: 32' , 'id']])
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Position based indexing
# print(df.iloc[0:3,0:30])
# will also work for single cells
# print(df.iloc[3,30])
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Name based indexing 1st for row 2nd for column
#print(df.loc[250:255,'id':'diagnosis'])
# also to view any column 
#print(df.loc[250:255,'id'])
# to view any row
#print(df.loc[250,:])
# to extract a single cell 2nd argument has to be a string
# print(df.loc[250,'id'])
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# to get mean of any column 
#print(df.loc[:,"id"].mean())
# we donot find the mean along the row because datatypes are different
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# To modify the data
# Sometimes NaN and NaT are present as string so dropna() wont work
# df.replace(["NaN", 'NaT'], np.nan, inplace = True)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# to drop columns , multiple columns can also be dropped
# if u dont give inplace it wont go away
# df.drop(['id'] , axis = 1 , inplace = True)
# df.drop(columns=['id', 'Unnamed: 32'] , inplace = True)
# to drop rows where the matrix contains the row number
# df.drop([0, 1 , 2] , inplace = True)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# To check the NaN rows
#print(df[pd.isnull(df).any(axis=1)])
# to drop all rows with NaN value
# df.dropna(inplace = True)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# to display the first n values
# print(df.head(10))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# To get the name of any column
# print(df.columns[1])
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# removing rows with unwanted data

def RowRemover(x = 0):
    index = []
    rowiterator = []
    number_of_columns = len(df.columns)
    for i in range(number_of_columns):
        index.append(i)
        
    for i , row in df.iterrows():
        rowiterator.append(i)
        
    #print(len(index))
    #print(len(rowiterator))
    '''
    if(df.iloc[8,10] == 1):
        print('true')
    else:
        print('false')
    '''
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
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# function for making labels then pass through one hot encoder of keras
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
# ---------------------------------------------------------------------------- 

'''
Go through the data sometimes instead of NaN only 0's may be present
'''


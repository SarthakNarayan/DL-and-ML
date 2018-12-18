import numpy as np
import os
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def gettingData():
    Data_dir = r"C:\Users\Lenovo\Desktop\training set"
    # Same name as the folder , cat is 0 and dog is 1
    Categories = ["Cat" , "Dog"]
    
    '''
    # An idea of what the code does
    c= 0
    for category in Categories:
        path = os.path.join(Data_dir , category)
        for image in os.listdir(path):
            c = c+1
            print(image)
            
    print(c) # Sum of images present in both the directory
    '''
    
    images = []
    labels = []
    img_size = 100
    for category in Categories:
        path = os.path.join(Data_dir , category)
        label = Categories.index(category)
        for image in os.listdir(path):
            # The image is being loaded as color image you can also load it in grayscale
            img_array = cv2.imread(os.path.join(path , image))
            # we resize because size of all images are not same
            img_array = cv2.resize(img_array , (100 , 100) )
            images.append(img_array)
            labels.append(label)
            
    # it is a list we have to convert it into a numpy array
    images = np.array(images)
    labels = np.array(labels)
    print(images.shape)
    print(labels)
    print(labels.shape)
    
    # now that we have our image data in array form we can perform all sklearn operations
    # 1st is shuffling the input data
    X_train , Y_train = shuffle(images , labels , random_state = 12)
    print("After reshuffling the data")
    print(Y_train)
    print(X_train.shape)
    print(Y_train.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(X_train , Y_train ,
                                                        test_size = 0.2, 
                                                        random_state = 12)
    print("After splitting the data")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    # Now the data is ready to be fed to CNN
    # data can be either normalized using sklearn or keras

# Main function
if __name__ == '__main__':
    gettingData()

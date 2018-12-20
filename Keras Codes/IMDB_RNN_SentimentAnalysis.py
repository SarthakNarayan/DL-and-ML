from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, CuDNNLSTM, Dense
from keras.utils import np_utils
import numpy as np
from keras.models import load_model

'''
INFO ABOUT THE DATASET
If you look at the data you will realize it has been already pre-processed. 
All words have been mapped to integers and the integers represent the words
sorted by their frequency. This is very common in text analysis to represent
a dataset like this. So 4 represents the 4th most used word, 5 the 5th most
used word and so on... The integer 1 is reserved reserved for the start 
marker, the integer 2 for an unknown word and 0 for padding.
'''

# downloads the top 10000 words
vocabulary_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)

# Getting an idea of data
# Its either a positive(1) or a negative(0) sentiment
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#print('---review---')
#print(X_train[6])
#print('---label---')
#print(y_train[6])

# Note that the review is stored as a sequence of integers. 
# These are word IDs that have been pre-assigned to individual words
# We can use the dictionary returned by imdb.get_word_index() to map the 
# review back to the original words.
word2id = imdb.get_word_index()
# On printing we see word to id is a dictionary
# print(word2id)

def training_data_sentence(no = 6):
    id2word = {}
    for word , i in word2id.items():
        id2word[i] =  word
    #print(id2word)
    
    translation = []
    for i in X_train[no]:
        translation.append(id2word[i])
    #print(translation)
    print(y_train[no])

# to get the maximum length of training and testing array
def max_length():
    length = []
    for i in X_test:
        length.append(len(i))
    print("Max length for training data" , max(length))
    length = []
    for i in X_train:
        length.append(len(i))
    print("Max length for training data" , max(length))

'''
 In order to feed this data into our RNN, all input documents must have the same
 length. We will limit the maximum review length to max_words by truncating 
 longer reviews and padding shorter reviews with a null value (0). We can
 accomplish this using the pad_sequences() function in Keras
 The pad_sequences() function performs two operations. 
 
 First, movie reviews that have more than 500 words are truncated to 
 exactly 500 words in length by removing excess words from the beginning 
 of the review. You can remove excess words from the end of reviews by 
 specifying truncating='post'. Second, any movie review that has fewer 
 than 500 words is padded up to exactly 500 words by adding 0 values to 
 the beginning of the review. You can pad at the end of reviews by specifying
 padding='post'.
'''
# for now we set max word length to 500
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# Now run max_length and see both training and testing data have a max length of 500

n_classes = 2
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)

# 100 and 500 are more common embedding_size
embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
# the output of embedding layer is 500*32
model.add(CuDNNLSTM(100))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

model.fit(X_train , y_train , epochs=5  , validation_split=0.2)

test_loss , test_accuracy = model.evaluate(X_test , y_test)
print(test_loss , (test_accuracy)*100)

# saving the model
print("Saving model to disk \n")
mp = r"E:\savedmodels\imdb_model.h5"
model.save(mp)
print("Model saved")

# Prediction part
print("Loading the model")
model_loaded = load_model(r'E:\savedmodels\imdb_model.h5')
print('model loaded')


print("New review: \'The movie was horrible\'")
review = "The movie was horrible"
# all the enteries have to be made in small letters hence we use lower
review = review.lower()

words = review.split()
#print(words)

review = []
for word in words:
  if word not in word2id: 
    print("word not found" , word)
    break
  else:
    review.append(word2id[word]) 
print(review)

review = np.array(review)
# since for padding shape has to be [[]]
review = np.reshape(review , (1,len(review)))
# Now the sequence has to be padded
review = sequence.pad_sequences(review, maxlen=max_words)
#print(converted_sentence)

prediction = model_loaded.predict(review)
print(prediction)

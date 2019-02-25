import sklearn
import pandas as pd
import numpy as np
import gensim
import logging
import os

train_data = "/home/s1710414/MachineLearning/minorResearch/assertionGenerator/libsignal-protocol-c-data_v2.csv"
test_data = "/home/s1710414/MachineLearning/minorResearch/assertionGenerator/c-compiler-data_v2.csv"
output_path = "/home/s1710414/MachineLearning/minorResearch/assertionGenerator/CNN_result.txt"
output_file = open(output_path,'w')

def tokenlizer(line):
    return [token for token in line.split(" ") if token != " "]

def read_input(input_data):
    for line in input_data:
        yield tokenlizer(line)

df = pd.read_csv(train_data) 
X_data = np.array(df.iloc[:, 0])
Y_data1 = np.array(df.iloc[:, 1])
Y_data2 = np.array(df.iloc[:, 2])
Y_data3 = np.array(df.iloc[:, 3])
Y_data4 = np.array(df.iloc[:, 4])
Y_data5 = np.array(df.iloc[:, 5])

df_test = pd.read_csv(test_data) 
X_data_test = np.array(df_test.iloc[:, 0])
Y_data1_test = np.array(df_test.iloc[:, 1])
Y_data2_test = np.array(df_test.iloc[:, 2])
Y_data3_test = np.array(df_test.iloc[:, 3])
Y_data4_test = np.array(df_test.iloc[:, 4])
Y_data5_test = np.array(df_test.iloc[:, 5])

documents = list(read_input(X_data))

num_features = 30  # Word vector dimensionality 
# build vocabulary and train model
w2vmodel = gensim.models.Word2Vec(
    documents,
    size=num_features,
    window=10,
    min_count=0,
    workers=10)

w2vmodel.train(documents, total_examples=len(documents), epochs=10)

# save model to file
w2vmodel.wv.save_word2vec_format('model.bin', binary=True)

# Function to extract vector presentation for a line
def make_feature_vec(words, w2vmodel):
    nwords = 0

    feature_vec = np.zeros((num_features*len(words),),dtype="float32")  # pre-initialize (for speed)
  
    for word in words:
        nwords = nwords + 1.
        #feature_vec = np.add(feature_vec,w2vmodel.wv[word])
        feature_vec = np.concatenate((feature_vec,w2vmodel.wv[word]), axis=None)
    #if (nwords > 0):
       # feature_vec = np.divide(feature_vec, nwords)
    return feature_vec.T
  
def padding(X_vectors, max_lengh):
    features_matrix = [np.zeros((max_lengh,),dtype="float32") for _ in range(len(X_vectors))]
    for i in range (len(X_vectors)):
      feature_vec = np.empty(max_lengh)
      feature_vec[:len(X_vectors[i])] = X_vectors[i]
      feature_vec[len(X_vectors[i]):] = np.zeros((max_lengh - len(X_vectors[i]),),dtype="float32") 
      features_matrix[i] = feature_vec
    return features_matrix

def extract_features(X_data, w2vmodel):
    features_matrix = [np.zeros((num_features,),dtype="float32") for _ in range(len(X_data))]
    train_labels = np.zeros(len(X_data))
    
    max_lengh = 0
    for i in range (len(X_data)):
        if (len(X_data[i]) > 1):
            #if (i > 0):
             # words_pre = gensim.utils.simple_preprocess(X_data[i-1], deacc=False, min_len=1, max_len=100)
           # else:
             # words_pre = []
            words = gensim.utils.simple_preprocess(X_data[i], deacc=False, min_len=1, max_len=100)
          
            
            feature_vector =  make_feature_vec(words, w2vmodel)
            features_matrix[i] = feature_vector
            if max_lengh < len(feature_vector):
               max_lengh = len(feature_vector)
    
    return features_matrix, max_lengh
 

X_vectors, max_lengh = extract_features(X_data, w2vmodel)
print(max_lengh)
X_vectors_pad = padding(X_vectors, max_lengh)

X_vectors_test, max_lengh_test = extract_features(X_data_test, w2vmodel)
print(max_lengh_test)
X_vectors_pad_test = padding(X_vectors_test, max_lengh)

#Neural network
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Conv1D, MaxPooling1D 
from keras.wrappers.scikit_learn import KerasClassifier 
import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, accuracy_score, make_scorer

def classification_report_with_accuracy_score(y_true, y_pred):
    output_file.write(classification_report(y_true, y_pred) )
    return accuracy_score(y_true, y_pred)

def create_model(): 
  model = Sequential()
  model.add(Conv1D(128, 2, padding='valid', activation='relu', strides=1, input_shape=(max_lengh,1))) 
  model.add(MaxPooling1D(2)) 
  model.add(Flatten()) 
  model.add(Dense(64, activation='relu')) 
  model.add(Dropout(0.5)) 
  model.add(Dense(32)) 
  model.add(Dense(1, activation='sigmoid')) 
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
x = np.reshape(X_vectors_pad, (-1, max_lengh, 1)) 
y1 = np.reshape(Y_data1, (-1, 1))
y2 = np.reshape(Y_data2, (-1, 1))
y3 = np.reshape(Y_data3, (-1, 1))
y4 = np.reshape(Y_data4, (-1, 1))
y5 = np.reshape(Y_data5, (-1, 1))

x_test = np.reshape(X_vectors_pad_test, (-1, max_lengh, 1)) 
y1_test = np.reshape(Y_data1_test, (-1, 1)) 
y2_test = np.reshape(Y_data2_test, (-1, 1))
y3_test = np.reshape(Y_data3_test, (-1, 1))
y4_test = np.reshape(Y_data4_test, (-1, 1))
y5_test = np.reshape(Y_data5_test, (-1, 1))

#nested_score = cross_val_score(clf, x,y1, cv=4, scoring=make_scorer(classification_report_with_accuracy_score))
#print(nested_score) 
#print("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))

##clf.fit(x, y1)
#Y1_test_predic = clf.predict(x_test)
#print(classification_report_with_accuracy_score(y1_test, Y1_test_predic))

"""#print ("Y2")
output_file.write("\nY2\n")
nested_score = cross_val_score(clf, x,y2, cv=4, scoring=make_scorer(classification_report_with_accuracy_score))
#print(nested_score) 
output_file.write("\n")
output_file.write(str(nested_score))
#print("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
output_file.write("\n")
output_file.write("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))

clf.fit(x, y2)
Y2_test_predic = clf.predict(x_test)
output_file.write("\n")
output_file.write(str(classification_report_with_accuracy_score(y2_test, Y2_test_predic)))


output_file.write("\n")
output_file.write("Y3")
output_file.write("\n")
nested_score = cross_val_score(clf, x,y3, cv=4, scoring=make_scorer(classification_report_with_accuracy_score))
output_file.write(str(nested_score)) 
output_file.write("\n")
output_file.write("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
output_file.write("\n")

clf.fit(x, y3)
Y3_test_predic = clf.predict(x_test)
output_file.write(str(classification_report_with_accuracy_score(y3_test, Y3_test_predic)))
output_file.write("\n")
"""
output_file.write("Y4")
output_file.write("\n")
nested_score = cross_val_score(clf, x,y4, cv=4, scoring=make_scorer(classification_report_with_accuracy_score))
output_file.write(str(nested_score))
output_file.write("\n")
output_file.write("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
output_file.write("\n")

clf.fit(x, y4)
Y4_test_predic = clf.predict(x_test)
output_file.write(str(classification_report_with_accuracy_score(y4_test, Y4_test_predic)))
output_file.write("\n")

output_file.write("Y5")
output_file.write("\n")
nested_score = cross_val_score(clf, x,y5, cv=4, scoring=make_scorer(classification_report_with_accuracy_score))
output_file.write(str(nested_score)) 
output_file.write("\n")
output_file.write("Accuracy: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
output_file.write("\n")

clf.fit(x, y5)
Y5_test_predic = clf.predict(x_test)
output_file.write(str(classification_report_with_accuracy_score(y5_test, Y5_test_predic)))



output_file.close()



'''####################### AIT 690 - Natural Language Proccessing ##################################
THE OBJECTIVE:
The purpose of the project is the detection of hate speech on social media.
The main idea is differentiate hate statements from abusive statements. The accuracy obatined should be greater than
the baseline of 51.6% (Baseline Sonar)
The annotated corpus is available through ICWSM-2018 (Founta et al. 2018)


The second script generates the second clasifier word 2 vector(using pretrained Glove vector)[7]
and the 2 models were generated(SVM and ANN) are fitted with word2Vec features.

USAGE:

The program is called '2_DetectHate_Model3-4.py', and it should be run from the command line with 2 arguments.
1_cleanFile.csv 200

The first argument should be the name of the clean file with tweets
The second argument should be the size of the pretrained Glove Vector Selected (50,100,200)

Limitations: the program was built using Python 3.6.3 python version

Example to run the program:
python 2_DetectHate_Model3-4.py 1_cleanFile.csv 200

Example:

-----------------------------------------------------
-------------------SVM + Glove Weighted Vector-------
              precision    recall  f1-score   support

     Abusive       0.82      0.90      0.86      1233
        Hate       0.66      0.50      0.57       489

   micro avg       0.79      0.79      0.79      1722
   macro avg       0.74      0.70      0.71      1722
weighted avg       0.77      0.79      0.78      1722

Accuracy SVM:78.57
F1 Score Weighted:77.56


-----------------------------------------------------
-------------------ANNs + Glove Weighted Vector-------

Test score: 0.597, accuracy: 0.716
Training accuracy: 71.84% / Validation accuracy: 71.60%

ALGORITHM:

0. The program starts when the user enters by console the required arguments to run the '2_DetectHate_Model3-4.py' program.
    Those arguments should be 2 arguments (1_cleanFile.csv and SizeofVector (50,100,200)).
    Otherwise, the user will receive the next message: 'You need to input  2 arguments to run scorer.py'
1. Using sys.argv; those arguments are kept in variables fileName, keyFile , outFileName
2. A method to preprocess sentences was created. It removed stopwords, special charcaters, URLs, numbers
1. Using sys.argv; those arguments are kept in variables inputFileName, embDim
2. Variables and files were initialized
3. Reading clean input file into a panda dataframe 
4. Bag of words from corpus obtained
5. Keeping only Glove  Pre-Trained Vector from the corpus .
6. Calculation of Weighted Vector for each tweet = Word 2 Vec
7. split of dataset in 20% for test and 80% to training.
8. Third model was executed with Weighted Vectors features and  Classifier SVM
9. Accuracy and metrics of the third model was printed.
10. Fouth model with Weighted Vectors features and  ANNs
11. Accuracy and metrics of the fourth model was printed.

REFERENCES:
[1] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. 2018.
[2] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior.Dataset. https://github.com/ENCASEH2020/hatespeech-twitter. 2018.
[3] Pennington J, Socher R, D.Manning C. "GloVe: Global Vectors for Word Representation". Retrieved: December 7, 2018. https://nlp.stanford.edu/projects/glove/
[7] Chen M. Efficient Vector Representation For Documents Through Corruption. ICLR 2017

COURSE: AIT 690 - Natural Language Proccessing
AUTHOR’s NAME: Sara Villanueva
DATE: 08 December 2018
'''

import tensorflow as tf
import csv, json,  sys , re , math
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.metrics import f1_score

from operator import itemgetter, attrgetter, methodcaller

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Activation


from keras.callbacks import Callback
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import SGD

from keras import metrics
import os , codecs

# control to verify minimum number of arguments
if lenArg<3:
    print('You need to input 2 arguments to run 2_DetectHate_Model3-4.py')
    exit()
else:
# reading arguments from command promt
    inputFileName  = sys.argv[1]
    embDim         = sys.argv[2]

# Initialize variables and files

ID_dic=defaultdict()
data = []
ID_cod=[]
#embDim=200     # number of dimension - gloVe Vector
GloveFile      = 'glove.twitter.27B.'+str(embDim)+'d.txt'

inputFileName  = '1_cleanFile.csv'

# In[6]:


## Reading clean input file
## Data loading in a Panda dataframe 
df = pd.read_csv(inputFileName, sep=',')

df.loc[df["Label"]=='hateful', 'Label'] = 1
df.loc[df["Label"]=='normal' , 'Label'] = 2
df.loc[df["Label"]=='abusive', 'Label'] = 0

df=df.drop(columns=['Id'])
df=df.drop(df[df.Label == 2].index)
df=df.dropna()

df=df.infer_objects()

df_x=df['Text'] 
df_y=df['Label'].values


# In[7]:



## Get Bag of words from corpus

all_words = []  
for line in list(df_x):
    words = line.split()
    for word in words:
        all_words.append(word)  
fd = nltk.FreqDist(all_words)
fd['unknown'] = 1 # add word unknown 

print('Number of words', len(fd.keys()))

## Glove Pre-Trained Vectors
## Keeping only Glove Vector from the corpus
i=0
model = {}
with open (GloveFile, encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        splitLine = line.split()
        word      = splitLine[0]              
        if word in fd.keys():
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding            
        i=i+1
f.close
sys.stdout.flush()
print ("But only ",len(model)," words loaded!")


# In[8]:


##  Calculation of Weighted Vector for each tweet = Word 2 Vec

j=0
ModelMatrix = np.zeros(shape=(len(df.index),embDim))

for doc in list(df_x): # df_x[0:10]  
    doc_words = doc.split()
    freq_seq = nltk.FreqDist(doc_words)    
    tot_words = sum(freq_seq.values())
    tot_words_unk=0
    F_Vector=model['like']*0  #Initializing Vector
    for word in freq_seq:
      try:
            F_Vector=F_Vector+(model[word]* freq_seq[word] /tot_words)
      except KeyError:
            tot_words_unk=tot_words_unk+1
      pass    
    if tot_words_unk>0:
        F_Vector = F_Vector + model['unknown']*tot_words_unk/tot_words
    ModelMatrix[j]=F_Vector    
    j=j+1
    


# In[28]:


# Split data 20% test and 80% for training with sklearn

X_train, X_test, y_train, y_test = train_test_split(ModelMatrix, df_y,  test_size=0.20, random_state=23)

# running Third model with Weighted Vectors features and  Classifier SVM

classifier = linear_model.SGDClassifier(max_iter=1000)
clf=classifier.fit(X_train, y_train)
predicted = clf.predict(X_test)
Accuracy = np.mean(predicted == y_test)

import numpy as np
np.random.seed(1337)
from sklearn import linear_model
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.metrics import f1_score

print("-----------------------------------------------------")
print("-------------------SVM + Glove Weighted Vector-------")
print(metrics.classification_report(y_test, predicted, target_names={'Abusive','Hate'}))
print("Accuracy SVM:"+ str(round(Accuracy*100,2)))
print("F1 Score Weighted:"+ str(round(f1_score(y_test, predicted, average='weighted')*100,2)))


# In[24]:


#  Fouth model with Weighted Vectors features and  ANNs

y_train_C = to_categorical(np.asarray(y_train))
y_test_C = to_categorical(np.asarray(y_test))

size= len(df.index)
model = Sequential()
model.add(Dense(16, input_dim=embDim, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid'))
# Optimization selected
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[42]:


NUM_EPOCHS=30
BATCH_SIZE=10

model_hist=model.fit(X_train,y_train,validation_data=(X_test, y_test),batch_size=BATCH_SIZE,epochs=NUM_EPOCHS)


# In[40]:


score = model.evaluate(X_test, y_test, verbose=1)

print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))

print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*model_hist.history['acc'][-1], 100*model_hist.history['val_acc'][-1]))

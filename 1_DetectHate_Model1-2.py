'''####################### AIT 690 - Natural Language Proccessing ##################################
THE OBJECTIVE:
The purpose of the project is the detection of hate speech on social media.
The main idea is differentiate hate statements from abusive statements.
The annotated corpus is available through ICWSM-2018 (Founta et al. 2018)

This first script filters and clean original tweets.
The first BOW classifier is generated and the first 2 models(Naive Bayes and SVM) are fitted with those features.

USAGE:
The program is called '1_DetectHate_Model1-2.py', and it should be run from the command line with 3 arguments.
data_hatespeechtwitter.json hatespeechtwitter.tab 1_cleanFile.csv

The first argument should be the name of file with the original tweets
The second  argument should be the name of the file with annotated classification.(Founta et al. 2018)
The third argument should be the name of the outputfile with clen tweets

Limitations: the program was built using Python 3.6.3 python version

Example to run the program:
python 1_DetectHate_Model1-2.py data_hatespeechtwitter.json hatespeechtwitter.tab 1_cleanFile.csv

Example:
-----------------------------------------------------
-----------------MultinomialNB-----------------------
              precision    recall  f1-score   support

     Abusive       0.80      0.90      0.85      1233
        Hate       0.65      0.44      0.52       489

   micro avg       0.77      0.77      0.77      1722
   macro avg       0.72      0.67      0.69      1722
weighted avg       0.76      0.77      0.76      1722

Accuracy MultinomialNB:77.24
F1 Score Weighted:75.75
-----------------------------------------------------
----------------------SVM----------------------------
              precision    recall  f1-score   support

     Abusive       0.82      0.89      0.85      1233
        Hate       0.64      0.51      0.56       489

   micro avg       0.78      0.78      0.78      1722
   macro avg       0.73      0.70      0.71      1722
weighted avg       0.77      0.78      0.77      1722

Accuracy SVM:77.82
F1 Score Weighted:76.97


ALGORITHM:

0. The program starts when the user enters by console the required arguments to run the '1_DetectHate_Model1-2.py' program.
    Those arguments should be at least 3 arguments (data_hatespeechtwitter.json hatespeechtwitter.tab 1_cleanFile.csv).
    Otherwise, the user will receive the next message: 'You need to input  3 arguments to run scorer.py'
1. Using sys.argv; those arguments are kept in variables fileName, keyFile , outFileName
2. A method to preprocess sentences was created. It removed stopwords, special charcaters, URLs, numbers
3. Tag file was read and labels were kept in a dictionary.
4. Json File with tweets was read and it was kept it in an array.
5. A clean file was created with preprocess tweets and it was removed 'rt' words.
6. reading the clean file and it was removed tweets with 'Normal' label
7. initialization of method 'countVectorizer' to create feature bag of words. 
8. split of dataset in 20% for test and 80% to training.
9. First classifier 'Naivebayes' with BOW features execution and its predictions were generated. Accuracy and F1 score were calculated. 
10. Second classifier 'SVM' with BOW features execution and its predictions were generated. Accuracy and F1 score were calculated.

REFERENCES:
[1] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. 2018.
[2] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior.Dataset.  https://github.com/ENCASEH2020/hatespeech-twitter. 2018.
[6] scikit learn."sklearn. feature_extraction.text.CountVectorizer". Retrieved: December 7, 2018. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

COURSE: AIT 690 - Natural Language Proccessing
AUTHOR’s NAME: Sara Villanueva
DATE: 08 December 2018
'''

import csv, json,  sys , re , math
import pandas as pd
import numpy as np
import scipy

from collections import defaultdict
import collections, numpy
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE

lenArg= len(sys.argv)

# control to verify minimum number of arguments
if lenArg<3:
    print('You need to input 3 arguments to run 1_DetectHate_Model1-2.py')
    exit()
else:
# reading arguments from command promt
    fileName        = sys.argv[1]
    keyFile         = sys.argv[2]
    outFileName     = sys.argv[3]


# Initialize variables and files
ID_dic=defaultdict()
data = []
ID_cod=[]
#fileName='data_hatespeechtwitter.json' 
#keyFile='hatespeechtwitter.tab'
#outFileName ='1_cleanFile.csv'


# Function to clean and preprocess all tweets
def preprocess(raw_text):
    
    # keep only words    
    raw_text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", raw_text)
    raw_text = re.sub(r"http\S+", "", raw_text)
    raw_text = re.sub("[^a-zA-Z]", " ", raw_text) 
    raw_text = re.sub(r'[^\x00-\x7f]',r'', raw_text)

    # convert to lower case and split
    words = raw_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # remove special characters
    cleaned_word_list=""
    for item in meaningful_words:        
        item1= item.encode('utf-8','ignore')

        if cleaned_word_list=="":
            cleaned_word_list=item1.decode('unicode_escape')
        else:
            cleaned_word_list=cleaned_word_list+" "+item1.decode('unicode_escape')

    return cleaned_word_list


# Reading Tag File
with open(keyFile) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if line[1] in ("hateful" "abusive" "normal"): # Filter only 3 principal labels
                ID_dic[line[0]]= line[1]
                ID_cod.append(line[0])

# Reading Json File with all Tweets text
with open(fileName) as f:
    for line in f:
        data.append(json.loads(line))
        
# Creating a cvs output File with Ids and text
with open(outFileName,'w', encoding='utf8') as reportFile:
    reportFile.write("Id,Text,Label\n")
    for textC in data:
        textC_id = textC['id']
        if str(textC_id) in ID_cod:
            textC_pre = preprocess(textC['text'])
            if textC_pre[:2] =='rt' : textC_pre=textC_pre[-len(textC_pre)+3:] # get rid of 'rt' words
            reportFile.write(str(textC_id)+","+textC_pre+","+ ID_dic[str(textC_id)]+"\n")
reportFile.close
sys.stdout.flush()

#######################################

# Reading Clean Input File
df=pd.read_csv(outFileName, sep=',')

# Relabeling with codes
df.loc[df["Label"]=='hateful', 'Label'] = 1
df.loc[df["Label"]=='abusive', 'Label'] = 0
df.loc[df["Label"]=='normal', 'Label']  = 2

# Remove tweets with labels equal to 2 (normal)
df=df.drop(columns=['Id'])
df=df.drop(df[df.Label == 2].index)
df=df.dropna()


df=df.infer_objects()
df_x=df['Text'] 
df_y=df['Label'].values

# Getting features = Bag of words
cv = CountVectorizer()
X = cv.fit_transform(df_x)
print("Features: ", X.shape)

# Split data 20% test and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, df_y,  test_size=0.20, random_state=23)

# running first model with BOW features and  Classifier Naive Bayes
classifier = MultinomialNB()
clf        = classifier.fit(X_train, y_train)
predicted  = clf.predict(X_test)
Accuracy = np.mean(predicted == y_test)
print("-----------------------------------------------------")
print("-----------------MultinomialNB-----------------------")
print(metrics.classification_report(y_test, predicted, target_names={'Hate','Abusive'}))
print("Accuracy MultinomialNB:"+ str(round(Accuracy*100,2)))
print("F1 Score Weighted:"+ str(round(f1_score(y_test, predicted, average='weighted')*100,2)))

# running second model with BOW features and  Classifier SVM
classifier = linear_model.SGDClassifier(max_iter=1000)
clf        = classifier.fit(X_train, y_train)
predicted  = clf.predict(X_test)
Accuracy = np.mean(predicted == y_test)
print("-----------------------------------------------------")
print("----------------------SVM----------------------------")
print(metrics.classification_report(y_test, predicted, target_names={'Abusive', 'Hate'}))
print("Accuracy SVM:"+ str(round(Accuracy*100,2)))
print("F1 Score Weighted:"+ str(round(f1_score(y_test, predicted, average='weighted')*100,2)))

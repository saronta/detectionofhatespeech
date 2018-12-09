Project : Detection of Hate Speech on Social Media
----------------------------------------------------------------
Social platforms, such as Twitter, Facebook, are being misused to disseminate hate speech against some groups by race, sex, religion among others.
People with extreme ideologies are spreading online hate speech against some groups, which could generate a serious impact in the society. 
For that reason, differentiation between hate and abusive speech.
The purpose of the current project was to generate a predictive model to classify hate from abusive speech in tweets. 
An artificial neural network model was built on top of an annotated corpus of over 8000 tweets(Founta et al., 2018).  
Using word embedding to determine the best model, this differentiation model achieved an improvement of 28% over the baseline.  

The project followed 5 steps: dataset selection and calculation of baseline, data preprocessing, feature extraction, model execution and model evaluation
Two types of features were used: Bag of words and Word embeddings. And three type of classifiers were executed: NaiveBayes, SVM, ANNs.

Execution
-----------------------------------------------------
1) First execution:           1_DetectHate_Model1-2.py 
    python 1_DetectHate_Model1-2.py data_hatespeechtwitter.json hatespeechtwitter.tab 1_cleanFile.csv

2) Second execution:      2_DetectHate_Model3-4.py ( required file :  glove.twitter.27B.200d.txt )
   python 2_DetectHate_Model3-4.py 1_cleanFile.csv 200

Input Example:
-----------------------------------------------------
python 1_DetectHate_Model1-2.py data_hatespeechtwitter.json hatespeechtwitter.tab 1_cleanFile.csv
python 2_DetectHate_Model3-4.py 1_cleanFile.csv 200

Output Example:
-----------------------------------------------------
-----------------MultinomialNB +BOW---------
              precision    recall  f1-score   support

     Abusive       0.80      0.90      0.85      1233
        Hate       0.65      0.44      0.52       489

   micro avg       0.77      0.77      0.77      1722
   macro avg       0.72      0.67      0.69      1722
weighted avg       0.76      0.77      0.76      1722

Accuracy MultinomialNB:77.24
F1 Score Weighted:75.75
-----------------------------------------------------
----------------------SVM+BOW-----------------
              precision    recall  f1-score   support

     Abusive       0.82      0.89      0.85      1233
        Hate       0.64      0.51      0.56       489

   micro avg       0.78      0.78      0.78      1722
   macro avg       0.73      0.70      0.71      1722
weighted avg       0.77      0.78      0.77      1722

Accuracy SVM:77.82
F1 Score Weighted:76.97
-----------------------------------------------------
--------SVM + Glove Weighted Vector-------
              precision    recall  f1-score   support

     Abusive       0.82      0.90      0.86      1233
        Hate       0.66      0.50      0.57       489

   micro avg       0.79      0.79      0.79      1722
   macro avg       0.74      0.70      0.71      1722
weighted avg       0.77      0.79      0.78      1722

Accuracy SVM:78.57
F1 Score Weighted:77.56

-----------------------------------------------------
-------ANNs + Glove Weighted Vector-------

Test score: 0.597, accuracy: 0.716
Training accuracy: 71.84% / Validation accuracy: 71.60%


Prerequisites
----------------------------------------------------
python 		3.6.3
Conda 		4.5.11
tensorflow	1.9.0
keras		2.2.4

	Libraries
-----------------------------------------------------
	NLTK
	pandas
	numpy
	sklearn

	Dataset and Labels
-----------------------------------------------------
	Original Twitts		: data_hatespeechtwitter.json
	Labels Annotated		: hatespeechtwitter.tab
	Pretrained Glove Vectors	: glove.twitter.27B.200d.txt	

	Baseline
-----------------------------------------------------
	[5] SONAR API , Davidson T.


Outline of Solution
-----------------------------------------------------
-----------------------------------------------------
1) First execution:           1_DetectHate_Model1-2.py 
----------------------------------------------------------------
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

2) Second execution:      2_DetectHate_Model3-4.py
-----------------------------------------------------------------
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

Contributions (Papers)
-----------------------------------------------------
[1] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. 2018.
[2] Founta A, Djouvas C, Chatzakou D, Leontiadis I, Blackburn J, Stringhini G, Vakali A, Sirivianos M, Kourtellis N. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior.Dataset.  https://github.com/ENCASEH2020/hatespeech-twitter. 2018.
[3] Pennington J, Socher R, D.Manning C. "GloVe: Global Vectors for Word Representation". Retrieved: December 7, 2018. https://nlp.stanford.edu/projects/glove/
[4] Davidson T,Warmsley D, Macy M, Weber I.Automated Hate Speech Detection and the Problem of Offensive Language. 2017. 
[5] Davidson T,Warmsley D, Macy M, Weber I. Automated Hate Speech Detection and the Problem of Offensive Language. 2017. https://github.com/Hironsan/HateSonar/blob/master/README.md 
[7] Chen M. Efficient Vector Representation For Documents Through Corruption. ICLR 2017

Resources
-----------------------------------------------------
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/

Authors
-----------------------------------------------------
Sara Villanueva
Site: https://github.com/saronta

Acknowledgments
-----------------------------------------------------
Teacher Assistant: Kahyun Lee 
Professor: Hemant Purohit 
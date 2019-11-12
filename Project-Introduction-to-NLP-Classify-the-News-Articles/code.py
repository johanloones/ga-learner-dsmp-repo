# --------------
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


#Code Starts here
#Load Dataset
#Load data using path variable using .read_csv() method of pandas. Save it as news
news=pd.read_csv(path)

#All the columns are not relevant select only these two features TITLE (title of resource) 
#and CATEGORY (class label) and store it as news
news=news[['TITLE','CATEGORY']]

#Get the class distribution of class lables. You can get it using .value_counts() method on 
#CATEGORY column of news dataframe. Store it inside dist variable
dist=news['CATEGORY'].value_counts()

#Print out dist and news.head() to observe the class-wise distributions and the first observations respectively
print(dist,news.head())


# --------------
#Initialize stopwords as stop with set(stopwords.words('english'))
stop=set(stopwords.words('english'))


#To retain only alphabets for every instance, use a lambda function in combination with .apply() method that does so. 
#The function that you will be applying to every instance  will be re.sub("[^a-zA-Z]", " ",x). 
#This operation should be carried out on TITLE column only
display(news['TITLE'].head(25))
#for in news['TITLE']:
news['TITLE']=news['TITLE'].apply(lambda row:re.sub("[^a-zA-Z]", " ",row))
display(news['TITLE'].head(25))
#Use lambda function and .apply() method to convert the instances to lowercase and then tokenize (using .split()). 
#This operation should be carried out on TITLE column only
news['TITLE']=news['TITLE'].apply(lambda x:x.lower().split())

#Remove stopwords from every instance using a lambda function and .apply() method 
#retain only words which are in that instance but not in stop. 
#This operation should be carried out on TITLE column only
#news['TITLE']=news['TITLE'].apply(lambda x:[''.join(i) for i in x if i not in stop])
news['TITLE']=news['TITLE'].apply(lambda x:[i for i in x if i not in stop])

#The steps mentioned above gives a list for every instance across TITLE column. Join the list elements 
#into a single sentence using ' '.join() method of lists. Use both lambda function and .apply() method for it.
news['TITLE']=news['TITLE'].apply(lambda x:' '.join(x))

#Finally split into train and test using train_test_split function where feature is news["TITLE"], 
#target is news["CATEGORY"], test size is 20% and random state is 3. 
#Save the resultant variables as X_train, X_test, Y_train and Y_test
X_train, X_test, Y_train, Y_test=train_test_split(news['TITLE'],news['CATEGORY'],test_size=0.2,random_state=3)


# --------------
#Initialize Bag-of-words vectorizer using CountVectorizer() and TF-IDF vectorizer using TfidfVectorizer(ngram_range=(1,3)). 
#Save them as count_vectorizer and tfidf_vectorizer respectively
count_vectorizer=CountVectorizer()
tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,3))

#Fit each vectorizer on training and test features with text data and transform them to vectors.
#Bag-of-words 
X_train_count=count_vectorizer.fit_transform(X_train) 
X_test_count=count_vectorizer.transform(X_test)

#TF-IDF 
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train) 
X_test_tfidf=tfidf_vectorizer.transform(X_test)


# --------------
#Initialize two Multinomial Naive Bayes classifiers with MultinomialNB() and save them as nb_1 and nb_2. 
#for training and testing on both Bag-of-words and TF-IDF 
nb_1=MultinomialNB()
nb_2=MultinomialNB()

#Fit on train set
#Bag-of-words
nb_1.fit(X_train_count,Y_train)

#TF-IDF
nb_2.fit(X_train_tfidf,Y_train)

#Find Accuracy
#Bag-of-words
acc_count_nb=accuracy_score(nb_1.predict(X_test_count), Y_test)

#TF-IDF
acc_tfidf_nb=accuracy_score(nb_2.predict(X_test_tfidf), Y_test)

print(acc_count_nb, acc_tfidf_nb)


# --------------
import warnings
warnings.filterwarnings('ignore')

#Initialize two OneVsRestClassifier LogisticRegression classifiers 
#for training and testing on both Bag-of-words and TF-IDF 
logreg_1=OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_2=OneVsRestClassifier(LogisticRegression(random_state=10))

#Fit on train set
#Bag-of-words
logreg_1.fit(X_train_count,Y_train)

#TF-IDF
logreg_2.fit(X_train_tfidf,Y_train)

#Find Accuracy
#Bag-of-words
acc_count_logreg=accuracy_score(logreg_1.predict(X_test_count), Y_test)

#TF-IDF
acc_tfidf_logreg=accuracy_score(logreg_2.predict(X_test_tfidf), Y_test)

print(acc_count_logreg, acc_tfidf_logreg)

#Code ends here



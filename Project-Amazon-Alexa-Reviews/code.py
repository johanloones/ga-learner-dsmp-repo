# --------------
# Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Load Dataframe
df=pd.read_csv(path,sep='\t')

#Check the type of date.
print(df['date'].dtype)

#Convert the date attribute from string to datetime.date datatype
df['date']=pd.to_datetime(df['date'])

#Create a new column length and store the length of each of the verified_reviews in it.
df['length']=df['verified_reviews'].apply(lambda x:len(str(x)))




# --------------
# Plot the countplot for ratings where x = 'ratings', hue = feedback and data = df
sns.countplot(data=df,x='rating',hue='feedback')
plt.show()

# Plot the barplot for ratings where x = 'ratings',y = "variation" hue = feedback and data = df
sns.barplot(data=df,x='rating',y='variation',hue='feedback')
plt.show()



# --------------
# import packages
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize empty list 'corpus'
corpus=[]

# Stopwords
stop_words = set(stopwords.words("english"))

# for loop to fill in corpus
for review in df['verified_reviews']:
    # Retain alphabets
    review=re.sub('[^a-zA-Z]',' ',review)
    # Convert to lower case
    review=review.lower()
    review=review.split()
    # Tokenize
    #review=nltk.word_tokenize(review)
    # Initialize stemmer object
    ps=PorterStemmer()
    # Perform stemming
    review=[ps.stem(word) for word in review if word not in stop_words]
    # join elements of list
    review=' '.join(review)
    # add to 'corpus'
    corpus.append(review)
    
# display 'corpus'
display(corpus)


# --------------
# Import Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
cv=CountVectorizer(max_features=1500)

# Fit CountVectorizer on corpus & Independent variables
X=cv.fit_transform(corpus).toarray()

# Dependent variable
y=df['feedback']

# Counts
count=df['feedback'].value_counts()

# Split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)





# --------------
# Import Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Instantiate classifier
rf=RandomForestClassifier(random_state=2)

# Fit model on training data
rf.fit(X_train,y_train)

# Predict on test data
y_pred=rf.predict(X_test)

# Calculate the accuracy score
score=accuracy_score(y_test,y_pred)

# Calculate the precision
precision=precision_score(y_test,y_pred)

# Display 'score' and 'precision'
print('Accuracy score',score,'precision',precision)


# --------------
# Import Packages
from imblearn.over_sampling import SMOTE

# Instantiate smote
smote=SMOTE(random_state=9)

# fit_sample onm training data
X_train,y_train=smote.fit_sample(X_train,y_train)

# Fit model on training data
rf.fit(X_train,y_train)

# Predict on test data
y_pred=rf.predict(X_test)

# Calculate the accuracy score
score=accuracy_score(y_test,y_pred)

# Calculate the precision
precision=precision_score(y_test,y_pred)

# Display 'score' and 'precision'
print('Accuracy score after SMOTE',score,'precision after SMOTE',precision)



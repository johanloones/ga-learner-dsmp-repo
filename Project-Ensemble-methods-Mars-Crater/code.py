# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code Starts here
#Load dataset using pandas read_csv api in variable df,give file path as path.
df=pd.read_csv(path)

#Display first 5 columns of dataframe df.
print(df.head())

#Store all the features(independent values) in a variable called X
X=df.drop(columns='attr1089')

#Store the target variable (dependent value) in a variable called y
y=df.attr1089

#Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function. 
#Use test_size = 0.3 and random_state =4
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state =4)

#Initailaize MinMaxScaler and store it in a variable scaler
scaler=MinMaxScaler()

#Fit this scaler on the train data using .fit(X_train) and then transform both the train 
#and test features with .transform() method. Assign them back to X_train and X_test
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_test,X_train)
# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#Code Starts here
#Instantiate the Logistic regression model in a variable lr
lr=LogisticRegression()
import numpy as np

#Fit the model on the training data X_train and y_train
lr.fit(X_train,y_train)

#Store the predicted values based on X_test in a variable called y_pred
y_pred=lr.predict(X_test)

#Check the rocaucscore and store in the variable roc_score and print the same.
roc_score=roc_auc_score(y_test,y_pred)
print(roc_score)

#Code ends here


# --------------
##Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Code Starts here
#Intialize the Decision Tree model and store it in a variable dt .Use random_state=4
dt=DecisionTreeClassifier(random_state=4)

#Fit the model on the training data X_train and y_train
dt.fit(X_train,y_train)

#Store the predicted values based on X_test in a variable called y_pred
y_pred=dt.predict(X_test)

#Check the rocaucscore and store in the variable roc_score. Print the same.
roc_score=roc_auc_score(y_test,y_pred)
print(roc_score)

# Code ends here


# --------------
##Random Forest

from sklearn.ensemble import RandomForestClassifier


# Code Starts here
#Instantiate the Random Forrest model in a variable rfc. Use 'random_state=4`
rfc=RandomForestClassifier(random_state=4)

#Fit the model on the training data
rfc.fit(X_train,y_train)

#Store the predicted values based of X_test in a variable called y_pred
y_pred=rfc.predict(X_test)

#Check the rocaucscore and store in the variable roc_score and print the same.
roc_score=roc_auc_score(y_test,y_pred)
print(roc_score)

# Code ends here


# --------------
##Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code Starts here
#Use the BaggingClassifier() from sklearn to initialize a bagging classifier object. 
#Pass the parameter base_estimator= DecisionTreeClassifier(), n_estimators=100 , max_samples=100 
#and random_state=0, while initializing the object. Store the object in the variable 'bagging_clf'
bagging_clf=BaggingClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=100, max_samples=100, random_state=0)

#Use the fit() method of the bagging classifier object 'bagging_clf' on 'X_train' and 'y_train' 
#to train the models on the training data.
bagging_clf.fit(X_train,y_train)

#Use the score() method of the bagging classifier object 'bagging_clf' on 'X_test' and 'y_test' 
#to find out the accuracy of the test data and store the score in a variable called 'score_bagging'
score_bagging=bagging_clf.score(X_test,y_test)
print(score_bagging)

# Code ends here


# --------------
##Voting Classifier
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code Starts here
#Use the VotingClassifier() from sklearn to initialize a voting classifier object 
#pass the 'model_list' as input to the estimators parameter and 'hard' to the voting 
#parameter while initializing the object. Save the object in a variable 'voting_clf_hard'.
voting_clf_hard=VotingClassifier(estimators=model_list,voting='hard')

#Use the fit() method of the 'voting_clf_hard' to train the model on the 'X_train' and 'y_train'.
voting_clf_hard.fit(X_train,y_train)

#Use the score() method of the voting_clf_hard on 'X_test' and 'y_test' to find out 
#the accuracy of the test data and store it in a variable called 'hard_voting_score'
hard_voting_score=voting_clf_hard.score(X_test,y_test)
print(hard_voting_score)

# Code ends here



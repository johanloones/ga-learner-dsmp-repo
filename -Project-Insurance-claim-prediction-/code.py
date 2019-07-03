# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

##Data loading and splitting
# Code Starts here
#Load dataset using pandas read_csv api in variable df and give file path as path.
df=pd.read_csv(path)

#Display first 5 columns of dataframe df.
print(df.head())

#Store all the features(independent values) in a variable called X
X=df.drop(columns='insuranceclaim')

#Store the target variable (dependent value) in a variable called y
y=df['insuranceclaim']

#Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function. Use test_size = 0.2 and random_state = 6
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=6)

# Code ends here


# --------------
##Outlier Detection
#import packages
import matplotlib.pyplot as plt


# Code Starts here
#Plot the boxplot for X_train['bmi'].
sns.boxplot(X_train['bmi'])
plt.show()

#Set quantile equal to 0.95for X_train['bmi']. and store it in variable q_value.
q_value=X_train['bmi'].quantile(q=0.95)
print(q_value)

#Check the value counts of the y_train
y_train.value_counts()

# Code ends here


# --------------
##Correlation Check
# Code starts here
# Find the correlation between the features which are stored in 'X_train' and store the result in a variable called 'relation'.
relation=X_train.corr()

# Print relation.
print(relation)

# Plot pairplot for X_train.
sns.pairplot(X_train)
plt.show()
# Code ends here


# --------------
##Predictor check
# Code Starts here
#Create variable cols store the columns 'children','sex','region','smoker' in it.
cols=['children','sex','region','smoker']
#Create subplot with (nrows = 2 , ncols = 2) and store it in variable's fig ,axes
fig,axes=plt.subplots(nrows=2,ncols=2)

#Create for loop to iterate through row.
for i in range(2):
  #Create another for loop inside for to access column.
  for j in range(2):
    #Create variable col and pass cols[ i * 2 + j]
    col=cols[i*2+j]
    #Using seaborn plot the countplot where x=X_train[col], hue=y_train, ax=axes[i,j]
    sns.countplot(x=X_train[col],hue=y_train,ax=axes[i,j])
    
plt.show()
# Code ends here


# --------------
## Insurance claim prediction 
#Import Packages
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Code Starts here
#List of values for regularization parameters for logistic regression model.
# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

#Instantiate a logistic regression model with LogisticRegression() and save it to a variable called 'lr'.
lr=LogisticRegression()

#Inside GridSearchCV() pass estimator as the logistic model, param_grid=parameters. to do grid search 
#on the logistic regression model store the result in variable grid.
grid=GridSearchCV(estimator=lr,param_grid=parameters)

#Fit the model on the training data X_train and y_train.
grid.fit(X_train,y_train)

#Make predictions on the X_test features and save the results in a variable called 'y_pred.
y_pred=grid.predict(X_test)

#Calculate accuracy for grid and store the result in the variable accuracy and print accuracy.
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)

# Code ends here



# --------------
##Performance of a classifier
#Import Packages
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code Starts here
#Calculate the roc_auc_score and store the result in variable score.
score=roc_auc_score(y_test,y_pred, average='macro', sample_weight=None)

#Predict the probability using grid.predict_proba on X_test and take the second column and store the result in y_pred_proba.
y_pred_proba=grid.predict_proba(X_test)[:,1]

#Use metrics.roc_curve to calculate the fpr and tpr and store the result in variables fpr, tpr,.
fpr, tpr, threshold=metrics.roc_curve(y_test, y_pred, pos_label=None, sample_weight=None, drop_intermediate=True)

#Calculate the roc_auc score of y_test and y_pred_proba and store it in variable called roc_auc.
roc_auc=roc_auc_score(y_test,y_pred_proba, average='macro', sample_weight=None)

#Plot auc curve of 'roc_auc' using the line plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.legend(loc='upper_left')
plt.show()

# Code ends here



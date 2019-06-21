# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here

#Load dataset from 'path'(given) using "pd.read_csv()" in variable 'df' .
df=pd.read_csv(path)

#Display the first five columns of dataframe df.
print(df.head())

#Store all the features(independent values) in a variable called 'X'
X=df.drop(columns='Price')
print(X.head())
#Store the target variable (dependent value) in a variable called 'y'
y=df['Price']
print(y.head())

#Split the dataframe into X_train,X_test,y_train,y_test using the train_test_split() function. 
#Use test_size = 0.3 and random_state = 6
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)
print(X_train.head(),X_test.head(),y_train.head(),y_test.head())

#Find the correlation between the features that are stored in 'X_train' and store the result in a variable called 'corr'.
corr=X_train.corr()
#Print corr.
print(corr)


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code Starts here
#Instantiate a linear regression model with LinearRegression() and save it to a variable called 'regressor'.
regressor=LinearRegression()

#Fit the model on the training data 'X_train' and 'y_train'.
regressor.fit(X_train, y_train)

#Make predictions on the X_test features and save the results in a variable called 'y_pred'.
y_pred=regressor.predict(X_test)

#Find the r^2 score and store the result in a variable called 'r2'
r2=r2_score(y_test,y_pred)
print(r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
#Instantiate a lasso model with Lasso() and save it to a variable called 'lasso'.
lasso=Lasso()

#Fit the model on the training data X_train and y_train.
lasso.fit(X_train, y_train)

#Make predictions on the X_test features and save the results in a variable called 'lasso_pred'.
lasso_pred=lasso.predict(X_test)

#Find the r^2 score and store the result in a variable called 'r2_lasso'
r2_lasso=r2_score(y_test,lasso_pred)
print(r2_lasso)



# --------------
from sklearn.linear_model import Ridge

# Code starts here
#Instantiate a lasso model with Ridge() and save it to a variable called 'ridge'.
ridge=Ridge()

#Fit the model on the training data, X_train and y_train.
ridge.fit(X_train, y_train)

#Make predictions on the X_test features and save the results in a variable called 'ridge_pred'.
ridge_pred=ridge.predict(X_test)

#Find the r^2 score and store the result in a variable called 'r2_ridge'.
r2_ridge=r2_score(y_test,ridge_pred)
print(r2_ridge)

# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
#Initiate a LinearRegression() object and store it in a variable called 'regressor'.
regressor=LinearRegression()

#Calculate the cross_val_score on X_train,y_train having model = regressor and cv = 10, 
#and store the result in a variable called 'score'.
score=cross_val_score(regressor,X_train,y_train, cv = 10)

#Calculate the mean of 'score' and store it in variable 'mean_score'.
mean_score=np.mean(score)

#Print mean_score.
print(score,mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
#Initiate a pipeline for polynomial features as 'make_pipeline' having parameter 'PolynomialFeatures(2), LinearRegression()' 
#as its parameter.Store it in the 'model' variable .
model=make_pipeline(PolynomialFeatures(2), LinearRegression())

#Fit the model on the training data, X_train and y_train.
model.fit(X_train, y_train)

#Make predictions on the X_test features and save the results in a variable called 'y_pred'.
y_pred=model.predict(X_test)

#Find the r^2 score and store the result in a variable called 'r2_poly'
r2_poly=r2_score(y_test,y_pred)
print(r2_poly)



# --------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Code starts here
#The path for the dataset file has been store in variable path
#Load dataset using pandas read_csv api in variable df
df=pd.read_csv(path)

#Display first 5 rows of dataframe df.
print(df.head())

#Store all the features(independent values) in a variable called X
X=df.drop(columns=['list_price'])

#Store the target variable list_price (dependent value) in a variable called y
y=df['list_price']

#Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function.
#Use test_size = 0.3 and random_state = 6
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state = 6)
print(X_train.head(),X_test.head(),y_train.head(),y_test.head())
# code ends here



# --------------
import matplotlib.pyplot as plt

# Code starts here        
#Create variable cols store all the X_train columns in it.
cols=X_train.columns

#Create subplot with (nrows = 3 , ncols = 3) and store it in variable's fig ,axes
fig,axes=plt.subplots(nrows = 3 , ncols = 3)
#Create for loop to iterate through row.
for i in range(3):
  #Create a nested for loop to access column.
  for j in range(3):
    #Create variable col and pass cols[ i * 3 + j].
    col=cols[ i * 3 + j]
    #Plot the scatter plot of each column vs. the list_price
    axes[i,j].scatter(X_train[col],y_train)

# code ends here



# --------------
# Code starts here
#Find the correlation between the features which are stored in 'X_train' and store the result in a variable called 'corr'.
corr=X_train.corr()

#Print the correlation table
print(corr)

#Now from the above table find the features whose correlation is higher than (+/-)0.75
#We can see that the features of play_star_rating, val_star_rating and star_ratin have a correlation of greater than 0.75. 
#Remove play_star_rating and val_star_rating from X_train.
X_train.drop(columns=['play_star_rating', 'val_star_rating'],inplace=True)

#Remove play_star_rating and val_star_rating from X_test.
X_test.drop(columns=['play_star_rating', 'val_star_rating'],inplace=True)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
#Instantiate a linear regression model with LinearRegression() and save it to a variable called 'regressor'.
regressor=LinearRegression()

#Fit the model on the training data X_train and y_train.
regressor.fit(X_train,y_train)

#Make predictions on the X_test features and save the results in a variable called 'y_pred'.
y_pred=regressor.predict(X_test)

#Find the mean squared error and store the result in a variable called 'mse'. Print the value of mse.
mse=mean_squared_error(y_pred,y_test)
print(mse)

#Find the r^2 score and store the result in a variable called 'r2'. Print the value of r2.
r2=r2_score(y_test,y_pred)
print(r2)

# Code ends here


# --------------
# Code Starts here
#Calculate the residual for true value vs predicted value and store the result into a new variable 'residual'.
residual=y_test-y_pred

#Plot the histogram of the residual.
residual.hist()

# Code ends here



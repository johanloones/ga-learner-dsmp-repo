# --------------
##Data loading
#Importing header files
import pandas as pd
from sklearn.model_selection import train_test_split

# Code Starts here
#Load the dataset from 'path'(given) and store it in a variable called 'data'.
data=pd.read_csv(path)

#Create a subset of dataframe 'data' by dropping the columns ['customer.id','paid.back.loan'] and 
#save the subset in a variable called 'X'.
X=data.drop(columns= ['customer.id','paid.back.loan'])

#Store the target variable ('paid.back.loan') in a variable called y
y=data['paid.back.loan']

#Split 'X'and 'y' into X_train,X_test,y_train,y_test using train_test_split() function. 
#Use test_size = 0.3 and random_state = 0
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state = 0)
print(X_train,X_test,y_train,y_test)
# Code ends here


# --------------
##Target variable distribution
#Importing header files
import matplotlib.pyplot as plt

# Code Starts here
#Save the value counts of paid.back.loan(stored in y_train) in a variable called 'fully_paid' using "value_counts()".
fully_paid=y_train.value_counts()

#Plot a bar graph of 'fully_paid'.
fully_paid.plot.bar()

# Code ends here


# --------------
##Feature Split
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code Starts here
#From the column int.rate of 'X_train','X_test', remove the % character and convert the column into float.
#After that divide the values of int.rate with 100 and store the result back to the column 'int.rate'
X_train['int.rate']=X_train['int.rate'].apply(lambda x:x.rstrip('%')).astype(float)/100
X_test['int.rate']=X_test['int.rate'].apply(lambda x:x.rstrip('%')).astype(float)/100

 
#Create a subset of only numerical columns of 'X_train' using "select_dtypes()" and save it to a variable called 'num_df'
num_df=X_train.select_dtypes(include=np.number)

#Create a subset of only categorical(object) columns of 'X_train' using "select_dtypes()" and save it to a variable called 'cat_df'
cat_df=X_train.select_dtypes(include='object')
print(num_df.head(),cat_df.head())
# Code ends here


# --------------
##Numerical Features Visualisation
#Importing header files
import seaborn as sns

# Code Starts here
#Create a list called cols which is a list of all the column names of 'num_df'.
cols=num_df.columns

#Create subplot with (nrows = 9 , ncols = 1) and store it in variable's fig ,axes
fig,axes=plt.subplots(nrows=9,ncols=1,figsize=(15,15))

#Create for loop to iterate through rows.
for i in range(9):
  #Inside the loop, using seaborn, plot the boxplot where x=y_train, y=num_df[cols[i]] and ax=axes[i].
  sns.boxplot(x=y_train,y=num_df[cols[i]],ax=axes[i])
  
plt.show()
# Code ends here


# --------------
##Categorical Features Visualisation
# Code Starts here
#Create a list called cols which is a list of all the column names of 'cat_df'.
cols=cat_df.columns

#Create subplot with (nrows = 2 , ncols = 2) and store it in variable's fig ,axes
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(15,15))

#Create two for loops to access rows and columns
for i in range(2):
  for j in range(2):
    #Using seaborn plot the countplot where x=X_train[cols[i*2+j]], hue=y_train and ax=axes[i,j]
    sns.countplot(x=X_train[cols[i*2+j]],hue=y_train,ax=axes[i,j])

plt.show()

# Code ends here



# --------------
##Model Building
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code Starts here
#Iterate a for loop over categorical columns cat_df.
for i in cat_df.columns:
  #Fill the X_train null values with NA.
  X_train[i].fillna('NA',inplace=True)
  #Create a LabelEncoder object of sklearn and store it in a variable called 'le'
  le=LabelEncoder()
  #Use le to fit and transform all categorical columnscat_df of 'X_train'
  X_train[i]=le.fit_transform(X_train[i])
  #Fill the X_test null values with NA.
  X_test[i].fillna('NA',inplace=True)
  #Transform all categorical columns of 'X_test' using le.
  X_test[i]=le.transform(X_test[i])

#Replace No with 0 and Yes with 1 in both 'y_train' and 'y_test'
y_train.replace({'No':0,'Yes':1},inplace=True)
y_test.replace({'No':0,'Yes':1},inplace=True)

#Create a DecisionTreeClassifier object of sklearn with random_state=0 and save it in a variable called 'model'.
model=DecisionTreeClassifier(random_state=0)

#Fit 'X_train' and 'y_train' with 'model'
model.fit(X_train,y_train)

#Find the accuracy score of model with 'X_test' and 'y_test' and store it in a variable called acc
acc=model.score(X_test,y_test)
print(acc)
# Code ends here



# --------------
##Decision Tree Pruning
#Importing header files
from sklearn.model_selection import GridSearchCV

# Code Starts here
#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

#Create a DecisionTreeClassifier object of sklearn with random_state=0 and save it in a variable called 'model_2'.
model_2=DecisionTreeClassifier(random_state=0)

#Create a GridSearchCV object from sklearn object with estimator=model_2, param_grid=parameter_grid and cv=5
#and save it in a variable called 'p_tree'
p_tree=GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)

#Fit 'X_train' and 'y_train' with 'p_tree'
p_tree.fit(X_train,y_train)

#Find the accuracy score of p_tree with 'X_test' and 'y_test' and store it in a variable called acc_2
acc_2=p_tree.score(X_test,y_test)
print(acc_2)

# Code ends here


# --------------
##Tree visualising
#Importing header files
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code Starts here
#Create a export_graphviz object of sklearn with the hyper-parameters decision_tree=p_tree.best_estimator_, out_file=None, 
#feature_names=X.columns, filled = True and class_names=['loan_paid_back_yes','loan_paid_back_no'] and 
#save it in a variable called 'dot_data'
dot_data=export_graphviz(decision_tree=p_tree.best_estimator_,out_file=None,feature_names=X.columns,filled=True,
                         class_names=['loan_paid_back_yes','loan_paid_back_no'])

#To draw the decision-tree(graph), use "pydotplus.graph_from_dot_data()" and pass 'dot_data' as its parameter 
#and save the result in a variable called 'graph_big'
graph_big=pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here



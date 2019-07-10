# --------------
##Data loading and statistics

#Import Packages
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code Starts here
#The path for the data file has been store in variable path
#Load dataset using pandas read_csv api in variable dataset.
dataset=pd.read_csv(path)

#Display first 5 rows of dataframe dataset.
print(dataset.head())
dataset.drop(columns='Id',inplace=True)
#Check if there's any column which is not useful and remove it.
#Print out the statistical description of the above train dataset.
dataset.describe()

# Code ends here


# --------------
##Exploratory Data Analysis

# We will visualize all the attributes using Violin Plot - a combination of box and density plots
#Import Packages
import seaborn as sns
from matplotlib import pyplot as plt

#Code Starts here
#Store all the column names in the variable called as cols
cols=dataset.columns
#Store the length of number of features (excluding target variable) in the variable size respectively.
size=len(cols)-1
print(cols,size)

#Create variable x containing the target column.
x=dataset['Cover_Type']

#Create variable y containing all the columns exculding target.
y=dataset.drop(columns='Cover_Type')

for i in range(size):
    #Plot the violin plot for all attributes. 
    sns.violinplot(x=x,y=dataset[cols[i]])
    plt.show()
#Code ends here




# --------------
##Closer look towards relationship between different variables
#Import Packages
import numpy as np
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
#We can observe that the first 10 columns contain all continous features
#Select the first 10 features and store them in the variable subset_train
subset_train=dataset.iloc[:,:10]
print(subset_train.head())

#Calculate the Pearson correlation between these 10 features and store it in variable data_corr
data_corr=subset_train.corr()
print(data_corr)

#Plot a heatmap of data_corr using seaborn
sns.heatmap(data_corr)
plt.show()

#List the correlation pairs from data_corr using .unstack() and .sort_values(kind='quicksort') 
#& store it in variable correlation
correlation=data_corr.unstack().sort_values(kind='quicksort')
print(correlation)

#From correlation, using slicing, select the values above upper_threshold and 
#below lower_threshold but not equal to 1 and save it as corr_var_list.
#We neglect correlation value equal to 1 since this indicates correlation of a feature with itself.
corr_var_list=correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation!=1)]
print(corr_var_list)
# Code ends here




# --------------
##Data Cleaning , Data preparation
#Import libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

#Code Starts here
#Split the dataset in features and target and save the same in variables X and Y respectively
X=dataset.drop(columns='Cover_Type')
Y=dataset['Cover_Type']

#Split the data into chunks of 0.2 by using cross_validation class on X and Y and store it in X_train, X_test, Y_train, Y_test 
#and set the random_state to 0.
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Instantiate StandardScaler() to a variable called scaler. Perform the standard scaling on the continuous data on X_train and X_test 
#and store it in X_train_temp and X_test_temp.
scaler=StandardScaler()
numeric_cols=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']
cat_cols=['Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
X_train_temp=scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp=scaler.transform(X_test.iloc[:,:10])

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied 
#on it.
#Concatenate the scaled continuous data to categorical data above (X_train_temp and X_train) and store it in X_train1 
#and similarily concatenate ( X_test_temp and X_test) and store it in X_test1.
X_train1=np.concatenate((X_train_temp,X_train[cat_cols]),axis=1)
X_test1=np.concatenate((X_test_temp,X_test[cat_cols]),axis=1)

#Create a dataframe of rescaled data X_train1 which consists of columns and indexes set as of unscaled X_train 
#and store in the variable named as scaled_features_train_df
scaled_features_train_df=pd.DataFrame(X_train1,index=X_train.index,columns=X_train.columns)
#Create a dataframe of rescaled data X_test1 which consists of columns and indexes set as of unscaled X_test 
#and store in the variable named as scaled_features_test_df
scaled_features_test_df=pd.DataFrame(X_test1,index=X_test.index,columns=X_test.columns)

#Code ends here

#Code ends here


# --------------
##Feature Selection using Select percentile
#Import Packages
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

#Code Starts here
#Initialise a SelectPercentile() with the parameters score_func=f_classif & percentile=20 & store it in variable skb
skb=SelectPercentile(score_func=f_classif,percentile=20)

#Fit & transform skb on X_train1 and y_train and store the result in predictors.
predictors=skb.fit_transform(X_train1,Y_train)

#Access the scores generated by the above model using .scores_ attribute and store them as a list in the variable scores
scores=list(skb.scores_)

#Store all the feature names in a variable Features
Features=X_train.columns

#Create a dataframe named dataframe having Features and scores as the two columns.
dataframe=pd.DataFrame({'Features':Features,'scores':scores})

#Sort dataframe in descending order.
#Select the features that fall under top 20 percentile and store them as a list in top_k_predictors
#Print top_k_predictors
sort_dataframe=dataframe.sort_values(by='scores',ascending=False)
top_k_predictors=list(sort_dataframe['Features'][:predictors.shape[1]])
print(top_k_predictors)

#Code ends here




# --------------
##Effect of feature selection on model prediction
#Import Packages
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

#Code Starts here
#Define two variables as clf and clf1 intializing the one vs rest classifier with logistic regression 
#as the model as input.
model=LogisticRegression()
clf=OneVsRestClassifier(estimator=model)
clf1=OneVsRestClassifier(estimator=model)
#Create a variable model_fit_all_features which fits the classifier clf1 onto X_train and Y_train
model_fit_all_features=clf1.fit(X_train,Y_train)

#Predict the values with the above model fitted on X_test and store it in predictions_all_features.
predictions_all_features=clf1.predict(X_test)

#Get the accuracy score for the model above by comparing Y_test and predictions_all_features and store 
#it in the variable named as score_all_features.
score_all_features=accuracy_score(Y_test,predictions_all_features)

#Create a variable model_fit_top_features which fits the classifier clf onto scaled_features_train_df which 
#contains only top_k_ predictors thrown out by select percentile and Y_train
model_fit_top_features=clf.fit(scaled_features_train_df[top_k_predictors],Y_train)

#Predict the values with the above model fitted on scaled_features_train_df which contains only top predictors
#thrown out by select percentile and store it in predictions_top_features.
predictions_top_features=clf.predict(scaled_features_test_df[top_k_predictors])

#Get the accuracy score for the model above by comparing Y_test and predictions_top_features and store it in score_top_features.
score_top_features=accuracy_score(Y_test,predictions_top_features)

#Code ends here



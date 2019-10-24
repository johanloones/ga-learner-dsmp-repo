# --------------
import pandas as pd
from collections import Counter

#Code Starts here
# Load dataset
data=pd.read_csv(path)
display(data.isnull().sum())
display(data.describe())


# --------------
#Code Starts here
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

#Store the target column in the variable named as label
label=data['Activity']

#Plot a seaborn countplot with the x attribute as label
sns.countplot(x=label)

#Rotate the x labels to 90 degree
plt.xticks(rotation=90)
plt.show()

#Code ends here


# --------------
#Code Starts here
#Create a copy of the dataset and name it data_copy.
data_copy=data.copy()

#Create a empty column named as duration in the above copy of the dataset.
data_copy['duration']=''

#Create a grouby object duration_df grouping labels (WALKING_UPSTAIRS and WALKING_DOWNSTAIRS) 
#and subject and store the count in the above duration column also convert the duration_df to a pandas dataframe 
#and store it in the same.
mask=(label=='WALKING_UPSTAIRS') | (label=='WALKING_DOWNSTAIRS')
duration_df=data_copy.groupby([label[mask], 'subject'])['duration'].count() * 1.28 
duration_df=pd.DataFrame(duration_df)

#Create a variable plot_data which contains the results of duration_df in sorted style and the 
#labels WALKING_UPSTAIRS and WALKING_DOWNSTAIRS mapped to Upstairs and Downstairs respectively.
plot_data=duration_df.sort_values(by='duration').reset_index().replace({'WALKING_UPSTAIRS':'Upstairs','WALKING_DOWNSTAIRS':'Downstairs'})
display(plot_data)

#Create a seaborn barplot of subject vs duration for activity Upstairs and Downstairs.
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')

#Code ends here


# --------------
#Code Starts here
#Select the no. of continous features and store in variable feature_cols.
feature_cols=data.drop(['Activity','subject'],axis=1).columns
display(feature_cols)

#Store all the correlation scores of continous variables in the variable correlated_values , also stack 
#the data and convert into a dataframe by renaming the columns as such Feature_1, Feature_2 and Correlation_score.
correlated_values=data[feature_cols].corr().stack().reset_index()
correlated_values=pd.DataFrame(correlated_values)
correlated_values.rename(columns={'level_0':'Feature_1','level_1': 'Feature_2',0: 'Correlation_score'},inplace=True)
print(correlated_values)

#Create a column abs_correlation in the above variable to store absolute correlation values.
correlated_values['abs_correlation']=correlated_values['Correlation_score'].abs()
print(correlated_values)

#Sort the values on the basis of their absolute values and store it in s_corr_list.
s_corr_list=correlated_values.sort_values(by='abs_correlation')
print(s_corr_list)

#Store the correlated pairs exculding self correlated pairs in the variable named as top_corr_feilds with a 
#threshold of 0.8 for absolute correlation.
top_corr_fields=s_corr_list[(s_corr_list['abs_correlation']>0.8) & (s_corr_list['Feature_1']!=s_corr_list['Feature_2'])]
display(top_corr_fields)

#Code ends here


# --------------
#Code Starts here
#Importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import  accuracy_score

#Initiate the label encoder class and call it as le.
le=LabelEncoder()

#Transform the activity target column using label encoder and store it in the same.
data['Activity']=le.fit_transform(data['Activity'])

#Create a X and y having all the columns and the target feature respectively.
X=data.drop(columns='Activity')
y=data['Activity']

#Do the train test split with above X and y with a test size of 0.3 and random state of 40 and 
#store it in the variables X_train, X_test, y_train and y_test.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)

#Initiate the SVC class and call it as classifier.
classifier=SVC()

#Fit the SVC classifier on X_train and y_train and store it in clf.
clf=classifier.fit(X_train,y_train)

#Use clf to predict on X_test and save it as y_pred.
y_pred=clf.predict(X_test)

#Store the precision,accuracy and f1_score in variables precision,accuracy and f1_score respectively. 
#Use .error_metric() with parameter average = 'weighted' to find out these metrics.
precision, recall, f_score, support = error_metric(y_test, y_pred, average='weighted')

#Store the accuracy score of the model in the variable named as model1_score.
model1_score=accuracy_score(y_test,y_pred)

#Print out the accuracy, precision and f_score for the above model.
print('Precision\n',precision,'\nRecall\n', recall,'\nf1_score\n', f_score)

#Code ends here


# --------------
#Code Starts here
#Import Libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

#Intiate the linear svc class by fitting on X_train and y_train as follows (save it as lsvc) 
#C:0.01, penalty = 'l1', dual = False, random_state =42
lsvc=LinearSVC(C=0.01,penalty='l1',dual=False,random_state=42)
lsvc.fit(X_train,y_train)

#Initiate SelectFromModel class on lsvc and set prefit as True, also name the result of this variable as model_2.
model_2=SelectFromModel(lsvc, prefit=True)

#Create new_train_features and new_test_features using model_2 to transform on X_train and X_test respectively.
new_train_features=model_2.transform(X_train)
new_test_features=model_2.transform(X_test)

#Initiate the SVC class and call it as classifier_2.
classifier_2=SVC()

#Fit the SVC classifier on new_train_features and y_train and store it in clf_2.
clf_2=classifier_2.fit(new_train_features,y_train)

#Use clf_2 to predict on new_test_features and save it as y_pred_new.
y_pred_new=clf_2.predict(new_test_features)

#Store the accuracy score of the model in the variable named as model2_score.
model2_score=accuracy_score(y_test,y_pred_new)
precision, recall, f_score, support = error_metric(y_test, y_pred_new, average='weighted')

#Print out the accuracy, recall, precision and f_score for the above model
print('Precision\n',precision,'\nRecall\n', recall,'\nf1_score\n', f_score,'\nAccuracy\n',model2_score)

#Code ends here


# --------------
#Code Starts here
# Importing Libraries
from sklearn.model_selection import GridSearchCV

#Initiate a GridSearchCV object inside a variable selector, having SVC() as the primary model, accuracy 
#as the scoring attribute, and a variable parameters that will contain the below values for tuning your hyperparameters:
#'kernel': ['linear', 'rbf'] 'C': [100, 20, 1, 0.1]
#fit this model on new_train_features and y_train and print the best hyperparameters.
parameters={'kernel': ['linear', 'rbf'], 'C': [100, 20, 1, 0.1]}
selector=GridSearchCV(SVC(),parameters,scoring='accuracy')
selector.fit(new_train_features,y_train)

#Store the mean test score and standard deviation test score in means and stds also print the detailed grid 
#scores along with means , stds and params
print(selector.best_params_)
print(selector.cv_results_)
means=selector.cv_results_['mean_test_score']
stds=selector.cv_results_['std_test_score']
params=selector.cv_results_['params']
print(means,stds,params)


#Initialise a new SVC() class with the best hyperparameter values for 'kernel' and 'C' obtained above and 
#store it in the variable classifier_3.
classifier_3=SVC(C= 20, kernel= 'rbf')

#Fit this classifier on new_train_features and y_train and store it in clf_3.
clf_3=classifier_3.fit(new_train_features,y_train)

#Predict on the new_test_features using clf_3 and save the predictions inside y_pred_final.
y_pred_final=clf_3.predict(new_test_features)

#Store the accuracy score of the model in variable called model3_score.
model3_score=accuracy_score(y_test,y_pred_final)

precision,recall,f_score,support =error_metric(y_test,y_pred_final,average = 'weighted')

#Print out the accuracy, recall, precision and f_score for the above model
print('Precision\n',precision,'\nRecall\n', recall,'\nf1_score\n', f_score,'\nAccuracy\n',model3_score)

#Code ends here



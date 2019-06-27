# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code Starts here
#Data Loading
#Load the dataset from 'path'(given) and store it in a variable called 'data'.
data=pd.read_csv(path)

#Exploration Instructions & Observation
#Plot a histogram of Rating column to see the distribution of app ratings
#You can see from the plotted histogram that there exists Rating>5 which shouldn't be there
data['Rating'].plot.hist()

#Cleaning Instructions & Observation for Subset the dataframe 'data'
#Subset the dataframe 'data' such that it only contains rows where Rating is less than or equal to 5. 
#Save the subsetted dataframe back into 'data'.
data=data[data['Rating']<=5]

#Plot a histogram of Rating column again to see the distribution of app ratings
#Average rating of application in store is around 4 which is very high
data['Rating'].plot.hist()

#Code ends here


# --------------
# Code Starts here
# Null Value Treatment
  #Exploration Instructions
#Create a series containing count of null values of each column using "data.isnull().sum()" and 
#store it in a variable called 'total_null'
total_null=data.isnull().sum()

#Create a series containing percentage of null values out of total values of each column by 
#using "(total_null/data.isnull().count())" and store it in a variable called 'percent_null'
percent_null=(total_null/data.isnull().count())*100

#Concatenate 'total_null' and 'percent_null' along column axis(axis=1) with keys=['Total','Percent'] and 
#save it in a variable called 'missing_data'
missing_data=pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)

#Print 'missing_data'
print(missing_data)
#Observation
#There are null values in only two columns and that too in very few rows. We can straight off drop the null values.


#Cleaning Instructions
#Drop the null values from 'data'
data=data.dropna()

#After dropping, create two variables 'total_null_1' and 'percent_null_1' in a similar way to 'total_null' and 'percent_null'
total_null_1=data.isnull().sum()
percent_null_1=(total_null/data.isnull().count())*100

#Concatenate 'total_null_1' and 'percent_null_1' along column axis(axis=1) with keys=['Total','Percent'] and save it in a variable called 'missing_data_1'
missing_data_1=pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
#Print 'missing_data_1'
print(missing_data_1)
#Observation
#There are no null values left in the data

# Code ends here


# --------------

#Code Starts here
#Category vs Rating
#Exploration Instructions
#Using seaborn, plot the catplot where x="Category",y="Rating",data=data, kind="box" and height = 10.
sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)

#Rotate the x-axis labels by 90o
plt.xticks(rotation=90)

#Title the plot as Rating vs Category [BoxPlot]
plt.title('Rating vs Category [BoxPlot]')
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code Starts here
#Installs vs Ratings
#Exploration Instructions
#Print value counts of Installs column of 'data' to see its distribution
print(data['Installs'].value_counts())
#Observations
#The column Installs is not of type int
#The values have a large range(0- 500M)

#Cleaning Instructions
#Remove , and + from Installs column of 'data'.
#Convert the Installs column to datatype int
data['Installs'] = data['Installs'].apply(lambda x: x.rstrip('+').replace(',', '')).astype(int)

#Create a labelencoder object named 'le' using "sklearn" and transform the values of Installs column
le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
#Using seaborn, plot the regplot where x="Installs", y="Rating" and data=data.
sns.regplot(x="Installs", y="Rating", data=data)
#Title the plot as Rating vs Installs [RegPlot]
plt.title('Rating vs Installs [RegPlot]')
#Observations
#There is but a small positive correlation between number of installs and ratings

#Code ends here



# --------------
#Code Starts here
#Price vs Ratings
#Exploration Instructions
#Print value counts of Price column of 'data' to see its distribution
#Observations : The column Price is not of type int Majority of the values is 0(More than 90%)
print(data['Price'].value_counts())

#Cleaning Instructions
#Remove dollar sign from Price column of 'data'. 
#Convert the Price column to datatype float
data['Price']=data['Price'].str.lstrip('$').astype(float)

#Using seaborn, plot the regplot where x="Price", y="Rating" and data=data.
sns.regplot(x="Price", y="Rating", data=data)
#Title the plot as Rating vs Price [RegPlot]
plt.title('Rating vs Price [RegPlot]')
#Observations :There is a negative correlation between price and ratings. Higher price application may make customer disappointed, if they are not good enough.


#Code ends here


# --------------
#Code Starts here
#Genre vs Rating
#Exploration Instruction
#Try and see the unique values of the column Genres using "unique()"
print(data['Genres'].unique())
#Observations : The column Genres has 115 unique values Some of the apps have multiple genres hence resulting in 115 unique combinations

#Cleaning Instructions
#For better …, let's keep only one Genre per app. Split the values of column Genres by ; character and store only the first genre back in Genres column.
data['Genres']=data['Genres'].apply(lambda x: x.split(';')[0])

#Group Genres and Rating by Genres pass as_index=False and store its mean in a variable called 'gr_mean'
#gr_mean=data.groupby('Genres')['Genres','Rating'].mean()
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
#Print the statistics of 'gr_mean' using describe()" function
print(gr_mean.describe())

#Sort the values of 'gr_mean' by Rating using "sort_values()" function and save it back to 'gr_mean'
gr_mean=gr_mean.sort_values(['Rating'])

#Print the first and last value of gr_mean
#print(gr_mean.loc['Dating','Rating'],gr_mean.loc['Events','Rating'])
print(gr_mean['Rating'].iloc[0])
print(gr_mean['Rating'].iloc[-1])
#Observations
#The lowest of average rating on genres (Dating) is 3.97
#The highest of average rating on genres (Events) is 4.43
#Owing to Standard Deviation of 0.1, seems like genre doesn't have much effect on rating



#Code ends here


# --------------
#Code Starts here
#Last Updated vs Rating
#Exploration Instructions
#Print and visualise the values of Last Updated column of 'data'
print(data['Last Updated'].value_counts())	
#Observations : Last Updated is not of the date type.

#Cleaning Instructions
#Convert Last Updated to datetime format
data['Last Updated']=pd.to_datetime(data['Last Updated'])

#Find out the max value in Last Updated column and save it a variable called 'max_date'
max_date=pd.to_datetime(data['Last Updated']).max()

#Create new column Last Updated Days which is the difference between max_date and values of column Last Updated in days using "dt.days" function
data['Last Updated Days']=(max_date-data['Last Updated']).dt.days

#Using seaborn, plot the regplot where x="Last Updated Days", y="Rating", data=data
sns.regplot(x="Last Updated Days", y="Rating", data=data)

#Title the plot as Rating vs Last Updated [RegPlot]
plt.title('Rating vs Last Updated Days [RegPlot]')
#Observations : Higher the gap between the days last updated, lower the rating

#Code ends here



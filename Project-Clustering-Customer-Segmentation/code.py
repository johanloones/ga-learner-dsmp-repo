# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Load Offers
offers=pd.read_excel(path,sheet_name=0)

# Load Transactions
transactions=pd.read_excel(path,sheet_name=1)

#Add a new column 'n' to transactions and set the value as 1 across all observations. 
#This is done to identify customers who have purchased a particular offer
transactions['n']=1

# Merge dataframes
df=offers.merge(right=transactions)

# Look at the first 5 rows
display(offers.head())
display(transactions.head())
df.head()



# --------------
#Create a pivot table matrix using  on df with arguments 
#index='Customer Last Name', columns='Offer #' and values='n'
matrix=pd.pivot_table(data=df,index='Customer Last Name',columns=['Offer #'],values='n' )

#Since every customer did not purchase all the offers
#Fill in the missing values by 0s.
matrix.fillna(0,inplace=True)

#The last names of customers are on the index; translate it to a column using .reset_index(inplace=True)
#so that it is a column now
matrix.reset_index(inplace=True)

#Print out the first five rows of matrix now
display(matrix.head())



# --------------
# import packages
from sklearn.cluster import KMeans

#Initialize a KMeans object cluster 
cluster=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

#Create a new column 'cluster' in the dataframe matrix where you store the cluster centers for every observation from matrix. 
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])

#Print out first five rows using .head() method of matrix
display(pd.DataFrame(cluster.cluster_centers_))
display(matrix.head())


# --------------
from sklearn.decomposition import PCA

#Initialize a PCA object pca using PCA(n_components=2, random_state=0)
pca=PCA(n_components=2,random_state=0)

#Create a new column 'x' for matrix dataframe which denotes the X co-ordinates of every 
#observation in decomposed form using .fit_transform(matrix[matrix.columns[1:])[:,0] method of pca
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]

#Similarly create a new column 'y' which denotes the decomposed Y-co-ordinates of every observation
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

#Create a new dataframe clusters containing the column numbers 'Customer Last Name','cluster','x','y'.
clusters=matrix[['Customer Last Name','cluster','x','y']]

#Visualize clusters using scatter plot
clusters.plot.scatter(x='x',y='y',c='cluster',colormap='viridis')


# --------------
#First merge dataframes clusters and transactions  on Customer Last Name column and save it as data
data=clusters.merge(right=transactions)

#Merge offers and data and save it as data 
data=data.merge(right=offers)

#Initialize an empty dictionary champagne
champagne={}

counts_max=0
#Iterate using a for loop over the cluster numbers
for i in data['cluster'].sort_values(ascending=False).unique():
  print('iterator / cluster',i)
  #Create a new dataframe new_df
  new_df=data[data['cluster']== i]
  #Initialize a variable counts where you will sort the counts for every value of 'Varietal' column of new_df in a descending order
  counts=new_df['Varietal'].value_counts(ascending=False)
  counts_max=counts.max()
  print('First Entry counts',counts[0],'for iterator / cluster',i)
  #Now you will check if the  first entry of counts  index is 'Champagne'. 
  if (counts.index[0]=='Champagne'):
    print('champagne count',counts[0],'iterator / cluster',i)
    print('===='*20)
    champagne.update({i:counts[0]})
    
#Save the cluster number as cluster_champagne with the maximum value. 
cluster_champagne=max(champagne,key=champagne.get)

#Print out cluster_champagne to see which cluster purchases the most champagne!
print('cluster champagne',cluster_champagne)


# --------------
#Create an empty dictionary discount 
discount={}

#Iterate over all the clusters using a for loop and create a dataframe new_df for every cluster 
#within the loop
for i in data['cluster'].sort_values(ascending=False).unique():
  new_df=data[data['cluster']== i]
  #Calculate the average percentage of discounts for new_df by first adding all the discount values 
  #and then dividing the total by total number of observations of new_df. Store it in counts
  counts=new_df['Discount (%)'].sum()/len(new_df)
  #Add the cluster number as key and its value counts as value to dictionary discount
  discount.update({i:counts})
#Now find out the cluster number with the maximum average discount percentage from cluster_discount 
#and save it as cluster_discount
cluster_discount=max(discount,key=discount.get)
print(cluster_discount)



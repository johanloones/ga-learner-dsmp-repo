# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the dataframe
data=pd.read_csv(path)

#Value counts of loan_status 
loan_status=data['Loan_Status'].value_counts()

# BarPlot of loan_status 
loan_status.plot(kind='bar')


# --------------
#Code starts here

#Group the 'data' dataframe by Property_Area and Loan_Status and store it in a variable called 'property_and_loan'
property_and_loan=data.groupby(['Property_Area','Loan_Status'])

#Use the .size() method on 'property_and_loan' and then use .unstack() and save it back to 'property_and_loan'
property_and_loan=property_and_loan.size().unstack()

#Plot an unstacked bar plot of property_and_loan (It is similar to creating a stacked bar plot except change the parameter 'stacked' to False)
property_and_loan.plot(kind='bar',stacked=False)

#Name the x-axis as Property Area
plt.xlabel('Property Area')

#Name the y-axis as Loan Status
plt.ylabel('Loan Status')
# Code Ends here


# --------------
#Code starts here

#Group the 'data' dataframe by Education and Loan_Status and store it in a variable called 'education_and_loan'
education_and_loan=data.groupby(['Education','Loan_Status'])

#Use the .size() method on 'education_and_loan' and then use .unstack() and save it back to 'education_and_loan'
education_and_loan=education_and_loan.size().unstack()

#Plot an stacked bar plot of education_and_loan
education_and_loan.plot(kind='bar',stacked=True)

#Name the x-axis as Education Status
plt.xlabel('Education Status')

#Name the y-axis as Loan Status
plt.ylabel('Loan Status')

#Rotate the labels of x-axis by 45o
plt.xticks(rotation=45)

# Code Ends here


# --------------
#Code starts here
#Create a dataframe called 'graduate' which is a subset of 'data' dataframe with the condition "data['Education'] == 'Graduate'"
graduate=data[data['Education']=='Graduate']

#Create a dataframe called 'not_graduate' which is a subset of 'data' dataframe with the condition "data['Education'] == 'Not Graduate'"
not_graduate=data[data['Education']=='Not Graduate']

#Plot a density plot LoanAmount of 'graduate' dataframe using "Series.plot()" and pass the parameter kind='density' and label='Graduate'
graduate.LoanAmount.plot(kind='density',label='Graduate')

#Do the same for LoanAmount of 'not_graduate' dataframe but with label='Not Graduate'
not_graduate.LoanAmount.plot(kind='density',label='Not Graduate')

#Code ends here

#For automatic legend display
plt.legend()












#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here

#Create three subplots with (nrows = 3 , ncols = 1) and store it in variable's fig ,(ax_1,ax_2,ax_3)
fig,(ax_1,ax_2,ax_3)=plt.subplots(nrows=3,ncols=1)

#Since both are continuous variables, plot scatter plot between 'ApplicantIncome' and LoanAmount using ax_1. Set axis title as Applicant Income
ax_1.scatter(x=data['ApplicantIncome'],y=data['LoanAmount'])
ax_1.set_title('Applicant Income')

#Plot scatter plot between 'CoapplicantIncome' and LoanAmount using ax_2. Set axis title as Coapplicant Income
ax_2.scatter(x=data['CoapplicantIncome'],y=data['LoanAmount'])
ax_2.set_title('Coapplicant Income')

#Create a new column in the dataframe called 'TotalIncome' which is a sum of the values of columns ApplicantIncome and CoapplicantIncome
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']

#Plot scatter plot between 'TotalIncome' and LoanAmount using ax_3. Set axis title as Total Income
ax_3.scatter(x=data['TotalIncome'],y=data['LoanAmount'])
ax_3.set_title('Total Income')

# Code Ends here



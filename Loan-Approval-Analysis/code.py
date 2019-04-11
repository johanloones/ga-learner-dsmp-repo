# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

# code starts here
#The path for the dataset has been store in a variable path

#Create dataframe bank by passing the path of the file
bank=pd.read_csv(path)
print(bank)

#Create the variable 'categorical_var' and using 'df.select_dtypes(include = 'object')' check all categorical values.

categorical_var=bank.select_dtypes(include='object')
#print 'categorical_var'
print(categorical_var)

#Create the variable 'numerical_var' and using 'df.select_dtypes(include = 'number')' check all categorical values.
numerical_var=bank.select_dtypes(include='number')

#print 'numerical_var'
print(numerical_var)

# code ends here


# --------------
# code starts here

#From the dataframe 'bank', drop the column 'Loan_ID' to create a new dataframe 'banks'

banks=bank.drop('Loan_ID',axis=1)
print(banks)
#To see the null values, use "isnull().sum()" function and print it.
print(banks.isnull().sum())
#Calculate mode for the dataframe 'banks' and store in 'bank_mode'
bank_mode=banks.mode()
print(bank_mode)
#Fill missing(NaN) values of 'banks' with 'bank_mode' and store the cleaned dataframe back in 'banks'.
banks=banks.fillna(bank_mode.iloc[0])
#banks=banks.fillna(banks.mode().iloc[0])

print(banks)

#Check if all the missing values (NaN) are filled.
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
#We will use previously created dataframe banks for this task
#Generate a pivot table with index as 'Gender', 'Married', 'Self_Employed' and values as 'LoanAmount' , using mean aggregation

#avg_loan_amount=banks.pivot_table(index=['Gender', 'Married', 'Self_Employed'],values='LoanAmount').agg('mean')

avg_loan_amount=banks.pivot_table(index=['Gender', 'Married', 'Self_Employed'],values='LoanAmount',aggfunc='mean')
#Store the result in a variable called 'avg_loan_amount'
print(avg_loan_amount)
# code ends here



# --------------
# code starts here
#We will use the previously created dataframe banks for this task
#Create variable 'loan_approved_se' and store the count of results where Self_Employed == Yes and Loan_Status == Y.
loan_approved_se=len(banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')])
#Create variable 'loan_approved_nse' and store the count of results where Self_Employed == No and Loan_Status == Y.
loan_approved_nse=len(banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')])
#Loan_Status count is given as 614.
Loan_Status_count=614
print(loan_approved_se,loan_approved_nse)
#Calculate percentage of loan approval for self employed people and store result in variable 'percentage_se'.
percentage_se=(loan_approved_se*100)/Loan_Status_count
#Calculate percentage of loan approval for people who are not self-employed and store the result in variable 'percentage_nse'.
percentage_nse=(loan_approved_nse*100)/Loan_Status_count
print(percentage_se,percentage_nse)
# code ends here


# --------------
# code starts here
#Use "apply()"function to convert Loan_Amount_Term which is in months to year and store the result in a variable 'loan_term'.
loan_term=banks['Loan_Amount_Term'].apply(lambda x: x/12)

print(loan_term)
#Find the number of applicants having loan amount term greater than or equal to 25 years and store them in a variable called 'big_loan_term'.
big_loan_term=len(loan_term[loan_term >= 25])

print(big_loan_term)



# code ends here


# --------------
# code starts here
#Groupby the 'banks' dataframe by Loan_Status and store the result in a variable called 'loan_groupby'
loan_groupby=banks.groupby('Loan_Status')
print(loan_groupby)
#Subset 'loan_groupby' to include only ['ApplicantIncome', 'Credit_History'] and store the subsetted dataframe back in 'loan_groupby'
mean_values=loan_groupby[['ApplicantIncome', 'Credit_History']].mean()
#Then find the mean of 'loan_groupby'and store the result in a new variable 'mean_values'
print(mean_values)

# code ends here



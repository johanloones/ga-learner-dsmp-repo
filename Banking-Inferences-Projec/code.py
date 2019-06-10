# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
#Load the dataset from 'path'(given) and store it in a variable called 'data'.
data=pd.read_csv(path)

#Create a sample of 'data' using "sample()" with n=sample_size and random_state=0 and save it in a variable called 'data_sample'
data_sample=data.sample(sample_size,random_state=0)

#Store the mean of installment column of 'sample_data' in a variable called 'sample_mean'
sample_mean=data_sample['installment'].mean()

#Store the standard deviation of installment column of 'sample_data' in a variable called 'sample_std'
sample_std=data_sample['installment'].std()

#Find the margin of error using 'z_critical'(given),'sample_std' and 'sample_size' and save it in a variable called 'margin_of_error'
margin_of_error=z_critical*(sample_std/math.sqrt(sample_size))

#Find the confindence interval using 'sample_mean' and 'margin_of_error' and save it in a variable called 'confidence_interval'.
confidence_interval=sample_mean-margin_of_error,sample_mean+margin_of_error
print(confidence_interval)

#Store the mean of installment column of 'data' in a variable called 'true_mean'
true_mean=data['installment'].mean()
print(true_mean)

#Print and check if 'true_mean' falls in the range of 'confidence_interval
if (true_mean>sample_mean-margin_of_error) & (true_mean<sample_mean+margin_of_error):
  print(true_mean,'falls in the range of confidence_interval')

#Code Ends here  


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here

#An array of sample sizes to take('sample_size') is given

#Create subplot with (nrows = 3 , ncols = 1) and store it in variable's fig ,axes
fig ,axes=plt.subplots(nrows = 3 , ncols = 1)

#Create i loop with range (len(sample_size)).
for i in range (len(sample_size)):
  #Inside the i loop, initialise a list 'm' and create j loop with range (1000)(for no. of samples=`1000')
  m=[]
  for j in range(1000):
  #Inside the j loop create a dataset sample of data with sample_size==sample_size[i] using "sample()", 
  #find the mean of installment column of that sample and append it to list 'm'
    m.append(data.sample(sample_size[i],random_state=0)['installment'].mean())
  #Outside the j loop(but still inside the i loop), convert 'm' into a series called 'mean_series'
  mean_series=pd.Series(m)
  #Then, using axes[i], plot the corresponding histogram for mean_series
  axes[i].hist(mean_series)

#Code Ends here


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here

#Code starts here

#Small Business Interests
#hypothesis testing(one-sided) Null Hypothesis H_0: μ= value % and Alternate Hypothesis H_1: μ>value %
#From the column int.rate of 'data', remove the % character and convert the column into float.
#After that divide the values of int.rate with 100 and store the result back to the column 'int.rate'
data['int.rate']=data['int.rate'].map(lambda x: x.rstrip('%')).apply(lambda x: float(x)/100)
#Apply "ztest()" with x1 as data[data['purpose']=='small_business']['int.rate'] 
#and value as data['int.rate'].mean(), alternative='larger'(WHY?) and save the results in 'z_statistic' and 'p_value' respectively
z_statistic,p_value=ztest(x1=data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
#If 'p-value is less than 0.05, you can reject the null hypothesis, If 'p-value is greater than 0.05, 
#you can't reject the null hypothesis,
if p_value <0.05:
  print('Reject Null Hypothesis')
else:
  print('Accept Null Hypothesis')

#Code Ends here  


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here

#Installment vs Loan Defaulting 
#hypothesis testing(two-sided) Null Hypothesis H_0: loandefaulter mean yes = no Alternate Hypothesis H_1: loandefaulter yes != no
#Apply "ztest()" with x1 as data[data['paid.back.loan']=='No']['installment'] and 
#x2 as data[data['paid.back.loan']=='Yes']['installment'] 
#and save the results in 'z_statistic' and 'p_value' respectively
z_statistic,p_value=ztest(x1=data[data['paid.back.loan']=='No']['installment'],x2=data[data['paid.back.loan']=='Yes']['installment'])
print(z_statistic,p_value)
#If 'p-value is less than 0.05, you can reject the null hypothesis, 
#If 'p-value is greater than 0.05, you can't reject the null hypothesis,
if p_value <0.05:
  print('Reject Null Hypothesis')
else:
  print('Accept Null Hypothesis')

#Code Ends here  


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here

#Purpose vs Loan Defaulting
#Null Hypothesis : Distribution of purpose across all customers is same.
#Alternative Hypothesis : Distribution of purpose for loan defaulters and non defaulters is different.

#Create a variable 'yes' which is the value counts of purpose when paid.back.loan in 'data' is Yes
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()

#Create a variable 'no' which is the value counts of purpose when paid.back.loan in 'data' is No
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()

#Concat 'yes.transpose()'(transpose of 'yes') and 'no.transpose()'(transpose of 'no') along axis=1 with keys= ['Yes','No'] and store it in a variable called 'observed'
observed=pd.concat([yes.transpose(),no.transpose()],keys= ['Yes','No'],axis=1)
print(observed)
#Apply "chi2_contingency()" on 'observed' and store the result in variables named chi2, p, dof, ex respectively.
chi2, p, dof, ex=chi2_contingency(observed)

#Compare chi2 with critical_value(given)
#If chi-squared statistic exceeds the critical value, reject the null hypothesis that the 
#two distributions are the same, else null hypothesis cannot be rejected
if chi2 > critical_value:
  print('Reject Null Hypothesis')
else:
  print('Accept Null Hypothesis')

#Code Ends here  



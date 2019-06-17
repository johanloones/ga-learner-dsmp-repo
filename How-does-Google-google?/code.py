# --------------

# Code starts here
#The adjacency matrix adj_mat is provided to you.
import numpy as np
# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

#For this matrix perform the eigen vector decomposition using .linalg.eig() method of numpy. This function returns a tuple and save them as eigenvalues and eigenvectors
eigenvalues,eigenvectors=np.linalg.eig(adj_mat)

#The following is a single step divided into small steps:
#Find the eigen vector corresponding to 1 from eigenvectors (first column of eigenvectors), that is abs(eigenvectors[:,0])
#Normalize this by dividing with np.linalg.norm(eigenvectors[:,0],1). Save it as eigen_1
eigen_1= abs(eigenvectors[:,0])/np.linalg.norm(eigenvectors[:,0],1)

#Next save the most important page number by finding the index with highest value within eigen_1. This can be done by using the .where() method 
#from numpy Save it as page and print it out.
page=np.where(eigen_1==eigen_1.max())[0][0]+1
print(page)
# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
#The hyperlink matrix adj_mat is already defined for you. Initialize a stationary matrix init_I which has 1 at the first position and 0s in the rest 7 blocks of the numpy ndarray.
init_I=np.array([1,0,0,0,0,0,0,0])

#Use a for loop over 10 iterations where you update adj_mat according to the rule I^{k+1} = HI^k
# this can be done by .dot(adj_mat, init_I) . Also normalize init_I at every iteration using np.linalg.norm(init_I, 1)
# Perform iterations for power method
for i in range(10):
  init_I=np.dot(adj_mat, init_I)/np.linalg.norm(init_I, 1)

#Save the page number with highest importance as power_page. This can be found by .where() as done in the previous task.
power_page=np.where(init_I==init_I.max())[0][0]+1
print(power_page)

# Code ends here


# --------------
# Code starts here

#Problem with power method
#The new adjacency matrix this time for the new webpage connection structure shown in the above image. It is provided as new_adj_mat
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

#Initialize a stationary matrix new_init_I in the same manner as you did for the previous task
new_init_I=np.array([1,0,0,0,0,0,0,0])

#Use a for loop to iterate 10 times and update in the similar manner as you did for the previous task i.e. first take dot product np.dot(new_adj_mat, new_init_I) 
#and then normalize as done in previous task.
# Perform iterations for power method
for i in range(10):
  new_init_I=np.dot(new_adj_mat, new_init_I)/np.linalg.norm(new_init_I, 1)

#Print out new_init_I to check out its result. Observe how you get pagerank value for 3rd webpage as zero. Is it not possible as it has incoming connections.
print(new_init_I)


# Code ends here


# --------------
#Code Starts here
# Alpha value
alpha = 0.85

# Code starts here
# Modified adjancency matrix
#Initialize new hyperlink matrix G with the help of the mathematical formula given above. In the formula n can be taken as len(new_adj_mat)) 
#and 1 as np.ones(new_adj_mat.shape.Save it as G
n=len(new_adj_mat)
S=np.ones(new_adj_mat.shape)/n
I=np.ones(new_adj_mat.shape)
#G=αS+(1−α)*(1/n)*I
G=alpha*S+(I-alpha)*(1/n)

#Initialize stationary vector as final_init_I consisting of 1 at its beginning and rest all zeros in a 1D NumPy array
final_init_I=np.array([1,0,0,0,0,0,0,0])

#Perform 1000 iterations using a for loop to update the stationary vector in the same manner as for the Power Method. Also, do not forget to normalize it.
for i in range(1000):
  final_init_I=np.dot(G, final_init_I)/np.linalg.norm(final_init_I, 1)


#Print out final_init_I
print(final_init_I)


# Code ends here



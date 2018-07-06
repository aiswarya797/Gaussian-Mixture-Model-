"""
Step1: Get the dataset ready. Initial: We are taking 3 features into consideration (of IRIS).
	Label = 1 or 2 or 3 (for the three types respectively)

Step2: We use kmeans to get the initial guess of the means (=centroids)
       sigma would initialised to identical matrices (*hence diagonal, inter dependencies neglected)

Step3: Perform the E step, so that we assign the clusters to each of the points, maximising the likelihood.

Step4: Perform the M step, so that the parameters mu and sig are maximised.
Step5: We have to keep epsilon (=.0001 here), and the error (=square error here). find the shift, if less than the tolerance level, the method has converged.
"""

#import the necessary libraries

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint
from sklearn.cluster import KMeans


#The prob function calculates the Gaussian PDF:

def prob(val, mu, sig, lam):
	p = lam
	for i in range(len(val)):
		#i = int(i)
		p *= norm.pdf(val[i], mu[i], sig[i][i])

	return p
		
#Given the parameters, for each data in the dataset, we are finding prob that it is in the i'th cluster and assigning label with the maximum one.
def Expectation(data, parameters, lam):
	for i in range(data.shape[0]):
		#i = int(i)
		x1 = data['x1'][i]
		x2 = data['x2'][i]

				
		p_clus1 = prob([x1,x2], list(parameters['mu1']),list(parameters['sig1']),lam[0])
		p_clus2 = prob([x1,x2], list(parameters['mu2']),list(parameters['sig2']),lam[1])
		p_clus3 = prob([x1,x2], list(parameters['mu3']),list(parameters['sig3']),lam[2])
		"""print 'probabilities'
		print [p_clus1, p_clus2, p_clus3]"""		

		if p_clus1 > p_clus2:
			if p_clus1 > p_clus3:
				data['labels'][i] = 0
			else:
				data['labels'][i] = 2
		elif p_clus2> p_clus3:
				data['labels'][i] = 1
		else:
				data['labels'][i] = 2

	return data


def maximization(dataFrame, parameters,lam):
	points_assigned_to_cluster1 = dataFrame[dataFrame['labels'] == 0]
	points_assigned_to_cluster2 = dataFrame[dataFrame['labels'] == 1]
	points_assigned_to_cluster3 = dataFrame[dataFrame['labels'] == 2]

	"""print 'bla1'
	print points_assigned_to_cluster1
	print 'bla2'
	print points_assigned_to_cluster2
	print 'bla3'
	print points_assigned_to_cluster3"""

	percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
	percent_assigned_to_cluster2 = len(points_assigned_to_cluster2) / float(len(dataFrame))
	percent_assigned_to_cluster3 = len(points_assigned_to_cluster3) / float(len(dataFrame))	

	lam = [percent_assigned_to_cluster1, percent_assigned_to_cluster2, percent_assigned_to_cluster3]

	print lam

	parameters['mu1'] = [points_assigned_to_cluster1['x1'].mean(), points_assigned_to_cluster1['x2'].mean()]

  	parameters['mu2'] = [points_assigned_to_cluster2['x1'].mean(), points_assigned_to_cluster2['x2'].mean()]

	parameters['mu3'] = [points_assigned_to_cluster3['x1'].mean(), points_assigned_to_cluster3['x2'].mean()]

	print parameters['mu1']
	print parameters['mu2']
	print parameters['mu3']
  	
	parameters['sig1'] = [[points_assigned_to_cluster1['x1'].std(),0],[0,points_assigned_to_cluster1['x2'].std()]]

  	parameters['sig2'] = [[points_assigned_to_cluster2['x1'].std(),0],[0,points_assigned_to_cluster2['x2'].std()]]

	parameters['sig3'] = [[points_assigned_to_cluster3['x1'].std(),0],[0,points_assigned_to_cluster3['x2'].std()]]

	print parameters['sig1']	
	print parameters['sig2']
	print parameters['sig3']

	return [parameters, lam]



def distance(old_params, new_params, lam):
	dist = 0
	print [len(old_params), len(new_params)]
	for param in ['mu1', 'mu2', 'mu3']:
		#print old_params[param][0]
		for i in range(len(old_params)):
			dist += (old_params[param][i] - new_params[param][i]) ** 2
			#print dist
	return dist ** 0.5




#Dataset : IRIS
data = pd.read_csv('data_csv.csv')
x1 = data.iloc[:,0].values
x2 = data.iloc[:,1].values


y  = data.iloc[:,2].values


data = {'x1': x1, 'x2': x2,'labels': y}
df = pd.DataFrame(data=data)

# inspect the data
df.head()
df.tail()


#### Expectation- Maximization Algorithm

"""
The expectation step: It assumes mu and sigma and tries to maximize the likelihood of each data point, that is each data point is assigned to a cluster from which it most likely originated.
The maximization step: Maximizes the Gaussian parameters with maximum likelihood estimates."""


"""kmeans = KMeans(n_clusters = 3)

x_new = df.iloc[:,:-1].values
kmeans = kmeans.fit(x_new)
mu = kmeans.cluster_centers_"""

#sig = [np.eye(3)] * 3
mu = [[1,5], [4,9],[15,20]]
#mu = [[230,100,100], [100,230,100],[100,100,230]]             ----works perfectly!!!!!!!!!
sig = [ [1,0],[0,1] ]
#sig = [[1, 0, 0, 0, 0, 0, 0,0 ],[0, 1,0, 0, 0, 0, 0,0],[0, 0, 1, 0, 0, 0, 0,0],[0, 0, 0, 1, 0, 0, 0,0],[0, 0, 0, 0, 1, 0, 0,0],[0, 0, 0, 0, 0, 1, 0,0],[0, 0, 0, 0, 0, 0, 1,0],[0,0,0,0,0,0,0,1]]
guess = {
	'mu1' : mu[0],
	'sig1' : sig,
	'mu2' : mu[1],
	'sig2' : sig,
	'mu3' : mu[2],
	'sig3' : sig,
	#'lambda' : [.2,.3,.5]
	}
print [mu[0],mu[1],mu[2]]

lam = [.2,.3,.5]


df_copy = df.copy()

params = pd.DataFrame(guess)


print params


epsilon = .000001
shift = maxint
iters = 0


while shift > epsilon:
	iters += 1
	# E-step
	#print params
	updated_labels = Expectation(df_copy.copy(),params,lam)
	#print updated_labels

	# M-step
	updated_parameters = maximization(updated_labels, params.copy(),lam)
	#print updated_parameters

	# see if our estimates of mu have changed
	# could incorporate all params, or overall log-likelihood
	shift = distance(params, updated_parameters[0],updated_parameters[1])

	# logging
	print("iteration {}, shift {}".format(iters, shift))

	# update labels and params for the next iteration
	df_copy = updated_labels
	params = updated_parameters[0]
	lam = updated_parameters[1]
	
	#fig = plt.figure()
	#plt.scatter(data['x1'], data['x2'],data['x3'])

	#fig.savefig("iteration{}.png".format(iters))

	#print params['mu1']
	#print params['mu2']
	#print params['mu3']
	print lam

print updated_labels





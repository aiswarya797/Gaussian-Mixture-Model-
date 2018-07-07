import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint
from sklearn.cluster import KMeans

### Setup
# set random seed
rand.seed(42)

def prob(val, mu, sig, lam):
  p = lam
  for i in range(len(val)):
    p *= norm.pdf(val[i], mu[i], sig[i][i])
  return p


# assign every data point to its most likely cluster
def expectation(dataFrame, parameters):
  for i in range(dataFrame.shape[0]):
    x = dataFrame['x'][i]
    y = dataFrame['y'][i]
    p_cluster1 = prob([x, y], list(parameters['mu1']), list(parameters['sig1']), parameters['lambda'][0] )
    p_cluster2 = prob([x, y], list(parameters['mu2']), list(parameters['sig2']), parameters['lambda'][1] )
    if p_cluster1 > p_cluster2:
      dataFrame['label'][i] = 1
    else:
      dataFrame['label'][i] = 2
  return dataFrame


# update estimates of lambda, mu and sigma
def maximization(dataFrame, parameters):
  points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
  points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]
  percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
  percent_assigned_to_cluster2 = 1 - percent_assigned_to_cluster1
  parameters['lambda'] = [percent_assigned_to_cluster1, percent_assigned_to_cluster2 ]
  parameters['mu1'] = [points_assigned_to_cluster1['x'].mean(), points_assigned_to_cluster1['y'].mean()]
  parameters['mu2'] = [points_assigned_to_cluster2['x'].mean(), points_assigned_to_cluster2['y'].mean()]
  parameters['sig1'] = [ [points_assigned_to_cluster1['x'].std(), 0 ], [ 0, points_assigned_to_cluster1['y'].std() ] ]
  parameters['sig2'] = [ [points_assigned_to_cluster2['x'].std(), 0 ], [ 0, points_assigned_to_cluster2['y'].std() ] ]
  return parameters

# get the distance between points
# used for determining if params have converged
def distance(old_params, new_params):
  dist = 0
  for param in ['mu1', 'mu2']:
    for i in range(len(old_params)):
      dist += (old_params[param][i] - new_params[param][i]) ** 2
  return dist ** 0.5


"""
# 2 clusters
# not that both covariance matrices are diagonal
mu1 = [0, 5]
sig1 = [ [2, 0], [0, 3] ]

mu2 = [5, 0]
sig2 = [ [4, 0], [0, 1] ]

# generate samples
x1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T

xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
labels = ([1] * 100) + ([2] * 100)

data = {'x': xs, 'y': ys, 'label': labels}
df = pd.DataFrame(data=data)

# inspect the data
df.head()
df.tail()

fig = plt.figure()
plt.scatter(data['x'], data['y'], 24, c=data['label'])
fig.savefig("true-values.png")"""

#There are three classes in the data

data = pd.read_csv('IRIS.csv')
x = data.iloc[:,1].values
y = data.iloc[:,3].values           #considering two features only
labels = data.iloc[:,8].values      #labels

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(labels)

#y = y.reshape(-1,1)
labels = np.array(labels)

#onehotencoder = OneHotEncoder(categorical_features = [0])
#y = onehotencoder.fit_transform(y).toarray()

data = {'x': x, 'y': y,'label': labels}
df = pd.DataFrame(data=data)

# inspect the data
df.head()
df.tail()

fig = plt.figure()
#plt.scatter(data['x1'], data['x2'],data['x3'],data['labels'], 24, c=data['labels'])
plt.scatter(data['x'], data['y'])
plt.show();


### Expectation-maximization

"""# initial guesses - intentionally bad
guess = { 'mu1': [1,1],
          'sig1': [ [1, 0], [0, 1] ],
          'mu2': [4,4],
          'sig2': [ [1, 0], [0, 1] ],
          'lambda': [0.4, 0.6]
        }
"""

kmeans = KMeans(n_clusters = 3)
#x1 = df.iloc[:100, :-1].values
#x2 = df.iloc[101:, :-1].values
x = df.iloc[:,:-1].values
kmeans = kmeans.fit(x)
mu = kmeans.cluster_centers_
#sig = [np.eye(3)] * 3
#sig = [ [1,0,0],[0,1,0],[0,0,1]]*3 
sig = [[[1,0],[0,1]]]*3
########################################################################################

#three clusters, two features, means each mu[i] has two elements and there are three sets of mu[i], ie. 1<=i<=3
print mu
print mu[0]
print sig[0]

guess = {
	'mu1' : mu[0],
	'sig1' : sig[0],
	'mu2' : mu[1],
	'sig2' : sig[1],
	'mu3' : mu[2],
	'sig3' : sig[2],
	'lambda' : [.3,.3,.4]
	}

# probability that a point came from a Guassian with given parameters
# note that the covariance must be diagonal for this to work

# loop until parameters converge
shift = maxint
epsilon = 0.001
iters = 0
df_copy = df.copy()
# randomly assign points to their initial clusters
df_copy['label'] = map(lambda x: x+1, np.random.choice(2, len(df)))
params = pd.DataFrame(guess)

print df
print df.copy()
print df_copy.copy()
"""

while shift > epsilon:
  iters += 1
  # E-step
  updated_labels = expectation(df_copy.copy(), params)

  # M-step
  updated_parameters = maximization(updated_labels, params.copy())

  # see if our estimates of mu have changed
  # could incorporate all params, or overall log-likelihood
  shift = distance(params, updated_parameters)

  # logging
  print("iteration {}, shift {}".format(iters, shift))

  # update labels and params for the next iteration
  df_copy = updated_labels
  params = updated_parameters

  fig = plt.figure()
  plt.scatter(df_copy['x'], df_copy['y'], 24, c=df_copy['label'])
fig.savefig("iteration{}.png".format(iters))"""

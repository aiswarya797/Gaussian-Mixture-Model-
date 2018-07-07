"""The aim is: To use Histogram of Oriented Graphs to extract features, which is later fed into GMM for clustering. 
   For testing: we take 2 samples from each and calculate the responsibility vectors. If both give same results we conclude positive result.
   For accuracy part: we take all possible pairs from the dataset (computationally expensive) and calculate [The expected output, What output GMM gave], i.e. there would be 4 types,
   [same,same], [different, same], [same, different] and [different, different]. {[same,same] means the pair is from same cluster and GMM has put in the same cluster too. So our accuracy shall be:

   {#[same,same] + #[diff,diff]}/{ #[same,same] + #[diff,diff] + #[diff, same] + #[same, diff]}
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from skimage.feature import hog
from collections import defaultdict

def find_pairs(source,y):
        result = []
	res_with_label = []
	#print(len(source))
        for p1 in range(len(source)):
                for p2 in range(p1+1,len(source)):
                        result.append([source[p1],source[p2]])
			res_with_label.append([source[p1],y[p1],source[p2], y[p2]])
        return result, res_with_label

#For finding accuracy:

def find_hog(x1, x2):

	fd1 = hog(x1.reshape((28,28)), orientations = 15, pixels_per_cell= (2,2), block_norm = 'L2', cells_per_block = (1,1), visualize = False, multichannel = None)
	fd2 = hog(x1.reshape((28,28)), orientations = 15, pixels_per_cell= (2,2), block_norm = 'L2', cells_per_block = (1,1), visualize = False, multichannel = None)
	return [fd1,fd2]
	

#Dataset : MNIST (only 1000 samples, 100 from each class)

dataset = pd.read_csv('MNIST_1000.csv')

features = dataset.iloc[:, 1:785].values
labels = dataset.iloc[:,0].values

list_hog_fd = []

for feature in features:
	fd = hog(feature.reshape((28,28)), orientations = 15, pixels_per_cell= (2,2), block_norm = 'L2', cells_per_block = (1,1), visualize = False, multichannel = None)
	list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')
print  hog_features

gmm = GMM(n_components = 10).fit(hog_features)
label_gmm = gmm.predict(hog_features)

#print label_gmm

"""#########################feature = features[0]
fd = hog(feature.reshape((28,28)), orientations = 10, pixels_per_cell= (4,4), block_norm = 'L2', cells_per_block = (1,1), visualize = False, multichannel = None)
print fd
l = gmm.predict([fd])
print l

#fd = np.array(fd, 'float64') 
prob = gmm.predict_proba([fd])
print prob#############################################"""


"""for i in range(len(label_gmm)):
	print [label_gmm[i], labels[i]]"""


"""#For testing:
#Dataset : for testing : 2 samples from each class

test_data = pd.read_csv('test_MNIST.csv')
X = test_data.iloc[:,1:785].values
y = test_data.iloc[:,0].values

list_hog_test = []

for feature in X:
	fd = hog(feature.reshape((28,28)), orientations = 10, pixels_per_cell= (4,4), block_norm = 'L2', cells_per_block = (1,1), visualize = False, multichannel = None)
	list_hog_test.append(fd)
hog_features_test = np.array(list_hog_test, 'float64') 

probs = gmm.predict_proba(hog_features_test)
print probs.shape

for i in range(probs.shape[0]):
	print probs[i]
"""

"""
Finding accuracy:

1> find all pairs
2> find hog of all the features selected
3> find the responsibilities of all of them
4> using the already available labels we know if from same class or diff class
5> find same cluster/ diff cluster
6> find same/ diff from  GMM results.

"""


dataset_acc = pd.read_csv('acc_GMM_HOG.csv')
X = dataset_acc.iloc[:,1:785].values
y = dataset_acc.iloc[:,0].values

result, res_with_label = find_pairs(X,y)
#print result[360]
#print len(result)


list_of_pair_hog = []

for res in result:
	h1, h2 = find_hog(res[0], res[1])
	list_of_pair_hog.append([h1,h2])

"""print list_of_pair_hog[0]
list_of_pair_hog = np.array(list_of_pair_hog)
#list_of_pair_hog = list_of_pair_hog.reshape(1,-1)
prob1 = gmm.predict_proba([list_of_pair_hog[0][0]])
prob2 = gmm.predict_proba([list_of_pair_hog[0][1]])

print prob1
print prob2
"""

list_of_resp = []
list_of_pair_hog = np.array(list_of_pair_hog)
for l in list_of_pair_hog:
	
	prob1 = gmm.predict_proba([l[0]])
	prob2 = gmm.predict_proba([l[1]])
	if np.all(prob1 == prob2):
		list_of_resp.append(1)
	else:
		list_of_resp.append(0)


list_labels = []

for res in res_with_label:
	if(res[1] == res[3]):
		list_labels.append(1)
	else:
		list_labels.append(0)
	
sum_nr = 0;
for i in range(len(list_of_pair_hog)):
	if(list_of_resp[i] == 1):
		if(list_labels[i] == 1):
			sum_nr = sum_nr + 1

	if(list_of_resp[i] == 0):
		if(list_labels[i] == 0):
			sum_nr = sum_nr + 1	

print sum_nr
print len(list_labels)
print len(list_of_resp)

accuracy = (sum_nr)/(len(list_labels))

print accuracy
	







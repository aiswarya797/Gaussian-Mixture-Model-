"""This is an implementation of Gaussian Mixture Model using library in sklearn tool kit"""

from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import numpy as np


#Data
dataset = pd.read_csv('dog_eag.csv')

X = dataset.iloc[:, 1:784].values
y = dataset.iloc[:,786].values


gmm = GMM(n_components = 2).fit(X)
labels = gmm.predict(X)


print labels
print y
probs = gmm.predict_proba(X)
print (probs[:100].round(1))


#print gmm.means_
#print gmm.covariances_



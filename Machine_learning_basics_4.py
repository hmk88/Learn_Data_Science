# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:30:40 2019

@author: hhaq
"""

#Source: https://jakevdp.github.io/PythonDataScienceHandbook/ 
#################################Machine Learning basics


##################################Bayesian Classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)  ###Creates Gaussian blobs
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')


from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X,y);
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew= model.predict(Xnew)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim=plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

yprob=model.predict_proba(Xnew)
yprob[-8:].round(2)


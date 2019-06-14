# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:43:07 2019

@author: hhaq
"""

###########Source: https://jakevdp.github.io/PythonDataScienceHandbook/
#######Difference between GMM and K-means algorithm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

###########Let's see the weeknesses of k-means algorithm
####k-means does not have built-in way of accounting for oblong or eliptical cluster
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

######k-means draws circle at the center of each cluster 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

#####the data just muddled together if we stretch it
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

############Gaussian Mixture Model
from sklearn.mixture import GaussianMixture as GMM
gmm= GMM(n_components=4).fit(X)
labels=gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
probs= gmm.predict_proba(X)
print(probs[:5].round(3))
size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size);



#######Example of clustering data into an ellipse produce error
############for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_): AttributeError: 'GaussianMixture' object has no attribute 'covars_'


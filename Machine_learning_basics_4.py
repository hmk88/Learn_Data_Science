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

########################Multinominal Naive Bayesian classification
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names
data


categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model= make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels= model.predict(test.data)


from sklearn.metrics import confusion_matrix
mat= confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');



#####################Simple example of single string returning prediction


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('sending a payload to the ISS')
predict_category('discussing Islam vs atheism')
predict_category('discussing Buddhism vs Judiasm')
predict_category('determining the screen resolution')
predict_category('inches vs centimeter')

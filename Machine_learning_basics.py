# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:07:20 2019

@author: hhaq
"""
#Source: https://jakevdp.github.io/PythonDataScienceHandbook/ 
#################################Machine Learning basics



#########Linear regression model example


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = sns.load_dataset('iris')

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);

from sklearn.linear_model import LinearRegression 



model = LinearRegression(fit_intercept=True)
model


X = x[:, np.newaxis]
X.shape


model.fit(X, y)

model.coef_
model.intercept_



xfit = np.linspace(-1, 11)
xfit



Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit);

####################sklearn.model_selection instead of sklearn.cross_validation
from sklearn.model_selection import train_test_split

#######collecting the data
from sklearn.datasets import load_iris
iris= load_iris()

#######arranging the data
iris
X_iris, y_iris = iris.data, iris.target

X_iris
y_iris

X_iris.shape
y_iris.shape
Xtrain,Xtest,ytrain,ytest= train_test_split(X_iris, y_iris, random_state=1)

Xtrain
Xtest
ytrain
ytest

##########Now making prediction 
from sklearn.naive_bayes import GaussianNB  #######appropriate model class
model= GaussianNB() ####instantiate the model 
model.fit(Xtrain,ytrain)   #####fitting the model to the data 
y_model = model.predict(Xtest)    ##########predict the new data



from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


#############################################Unsupervised learning example 

from sklearn.decomposition import PCA #########appropriate model class
model = PCA(n_components=2) ###instantiating hyperparameter 
model.fit(X_iris)  ###fit the data
X_2D= model.transform(X_iris)  ####Transforming data into 2D
X_2D
#######################plotting the results
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);


#####################Iris clustering example 
from sklearn.mixture import GaussianMixture           ##model selection
model= GaussianMixture(n_components=3, covariance_type='full')   ##instantiating the hyperparameter
model.fit(X_iris)   ###fit the model
y_gmm= model.predict(X_iris)   ####predicting the cluster

############plooting the results
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, col='cluster', fit_reg=False);


#############Application: Character recogonition in an image 
####Pre-formatted digits in Scikit
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape




import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


X=digits.data
X.shape

y=digits.target
y.shape



###############Classification
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(Xtrain, ytrain)
y_model= model.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,y_model)

#############confusion matrix= where we have gone wrong 
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');














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











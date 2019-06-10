# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:38:32 2019

@author: hhaq
"""

#Source: https://jakevdp.github.io/PythonDataScienceHandbook/ 
#################################Machine Learning basics

##############Hyperparameters and model validation: Wrong way 

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

#######KNeighbor classifier
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(X,y)
y_model= model.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_model)



##############Hyperparameters and model validation: Right way 
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0,
                                                train_size=0.5)
model.fit(Xtrain,ytrain)
ytest_model=model.predict(Xtest)
accuracy_score(ytest, ytest_model)

###########Two validation trials
ytest_model= model.fit(Xtrain, ytrain).predict(Xtest)
ytrain_model = model.fit(Xtest,ytest).predict(Xtrain)
accuracy_score(ytest_model, ytest), accuracy_score(ytrain_model, ytrain)

################This section divides the data into five groups and cross validate it
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)




######################################Model selection 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degrees=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degrees),
                         LinearRegression(**kwargs))

import numpy as np

def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)


import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # plot formatting

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degrees in [1, 3, 5]:
    y_test = PolynomialRegression(degrees).fit(X,y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degrees))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best');




from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree', degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');



plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test);
plt.axis(lim);



    


# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:29:02 2019

@author: hhaq
"""

# Introduction to pandas library 
# http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/


import pandas as pd
import nympy as np
import matplotlib.pyplot as plot
p.set_option('max_columns', 100)
#matplotlib inline
 


##########################Series data structure analysis  

s=pd.Series([2, 3, 45, 'Vaasa'])
print (s)


d= {'Chicago': 1000, 'New york': 1300, 'Portland': 900, 'San francesco': 1100, 'Austin': 450, 'Boston': None}
cities= pd.Series(d)
print(d)

cities['Chicago']

cities[['New york', 'Portland', 'Austin']]

cities[cities >100]

print (cities > 100)
print ('\n')
print (cities[cities > 100])

cities['Chicago'] = 1200
cities ['Chicago']

cities[cities < 1000] = 500

print (cities [cities <1000])

print ('Washington DC' in cities)
print ('Austin' in cities)

cities /2
cities*2

np.square(cities)

print (cities [['Chicago', 'San francesco']])
print ('\n')
print (cities [['New york', 'Austin']])

print (cities [['Chicago', 'San francesco']] + cities [['New york', 'Austin', 'Chicago']])

cities.notnull()


print (cities.isnull())
print ('\n')
print (cities[cities.isnull()])






















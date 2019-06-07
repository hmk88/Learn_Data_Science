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
pd.set_option('max_columns', 100)
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


#################################DataFrame analysis


data = {'year': [2010, 2011, 2012, 2013, 2014, 2015, 2012, 2013],
        'team': ['Viking', 'Viking', 'Viking', 'Sparta', 'Sparta', 'Sparta', 'Roman', 'Roman'],
        'Wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'loses': [5, 8, 6, 1, 5, 10, 6, 12]}
football= pd.DataFrame(data, columns=['year', 'team', 'Wins', 'loses'])
football


from_csv = pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\data\\mariano-rivera.csv')
from_csv.head()



cols = ['num', 'game', 'date', 'team', 'home_away', 'opponent',
        'result', 'quarter', 'distance', 'receiver', 'score_before',
        'score_after']
no_headers = pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\data\\peyton-passing-TDs-2012.csv', sep=',', header=None,
                         names=cols)
no_headers.head()

football.to_excel('football.xlsx', index=False)

del football

football= pd.read_excel('football.xlsx', 'Sheet1')
football


#######################################URL connection 

url = 'https://raw.github.com/gjreda/best-sandwiches/master/data/best-sandwiches-geocode.tsv'

from_url = pd.read_table(url, sep='\t')
from_url.head(3)












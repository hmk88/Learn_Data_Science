# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:58:35 2019

@author: hhaq
"""

#Reference =    https://jakevdp.github.io/PythonDataScienceHandbook/03.01-introducing-pandas-objects.html



#####################################################Data series in detail
import numpy as np
import pandas as pd


data = pd.Series([2, 3, 4, 5])
data

data.values

data.index

data[2]

data[1:2]

data[1:3]


data[0:4]

data = pd.Series([2, 3, 4, 5], 
                 index=['a', 'b', 'c', 'd'])

data

data['c']

pd.Series(7, index=[100, 200, 300])


pd.Series({3:'c', 2:'b', 1:'a'})

pd.Series({3:'c', 2:'b', 1:'a'}, index=[2, 1])


###############################################33Dataframe in detail


population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

states= pd.DataFrame({'population': population,
                          'area': area})
    
states

states.index
states.columns











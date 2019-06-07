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

states['area']

pd.DataFrame(population, columns=['population'])



data= [{'a': i, 'b': i*2}
        for i in range(9)]
pd.DataFrame(data)


pd.DataFrame(np.random.rand(3,2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])


A=np.zeros(3)

pd.DataFrame(A)

print(A.size)

import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data

'c' in data

data.keys()

list(data.items())

data['b'] = 2
data


data['a':'c']

data[0:2]

data[(data > 0.1) & (data < 2)]




data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data.loc[1]


data.loc[1:3]

data.iloc[1]

data.iloc[1:3]


area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data


data['area']
data.area

data.area is data['area']

data.pop is data['pop']


data['density'] = data['area'] / data['pop']
data

data.values
data.T
data.values[0]

data.iloc[:3,:2]

data.loc[:'Illinois', :'pop']

data.ix[:'Illinois', :'pop']    

data.loc[data.density < 1, ['pop', 'density']]

data.iloc[1, 2] = 100
data

del data[2]

data

data['Florida':'Illinois']

data['Texas' : 'Florida']


















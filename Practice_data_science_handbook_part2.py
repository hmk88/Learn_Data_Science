# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:43:25 2019

@author: hhaq
"""

############################################Data operations


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser

df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
df


np.exp(ser)
np.sin(df*np.pi/4)


area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

population / area

area.index | population.index


A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])

A+B
A.add(B, fill_value=0)


A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))

B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))


A+B

fill = A.stack().mean()
A.add(B, fill_value= fill)

df = pd.DataFrame(A, columns=list('QRST'))

df - df.iloc[0]

df.subtract(df['R'], axis=0)


data
data.isnull()
data.notnull()
data[data.notnull()]
data.dropna()
data.dropna(axis='columns')
data.dropna(axis='columns', how='all')
data.dropna(axis='rows', thresh=3)
data
data.fillna(1)
data
data.fillna(method='ffill')
data.fillna(method='bfill')



###############################################Concatenate


x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]

np.concatenate([x,y,z])


ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])

pd.concat([ser1, ser2])

try:
    pd.concat([ser1, ser2], verify_integrity=True)
    
except ValueError as e:
    print('ValueError:', e)
    
ser1
ser2
ser1.append(ser2)


df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
    
df1
df2    
pd.merge(df1,df2)


df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print (df8, df9, pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))



pop = pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\data-USstates\\state-population.csv')
areas = pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\data-USstates\\state-areas.csv')
abbrevs = pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\data-USstates\\state-abbrevs.csv')


pop.head()
areas.head()
abbrevs.head()

merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')

merged

merged=merged.drop('abbreviation', 1)
merged.head()



######################################Aggregation 

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser


ser.sum()

ser.mean()


df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
df


df.mean()

df.mean(axis='columns')





rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
df

df.groupby('key').aggregate(['min', np.median, 'max'])



birth=pd.read_csv('C:\\Users\\hhaq\\Documents\\GitHub\\Learn_Data_Science\\births.csv')

birth.head()

birth['decade'] = 10 * (birth['year'] // 10)

birth.pivot_table(birth, index='decade', columns='gender', aggfunc='sum')
sns.set()
birth.pivot_table(birth, index='decade', columns='gender', aggfunc='sum').plot()



birth.pivot_table(birth, index='day',
                    columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean birth by day');

birth_by_date = birth.pivot_table(birth, 
                                    [birth.index.day])
birth_by_date.head()





date = pd.to_datetime("4th of July, 2015")
date

date.strftime('%A')

date + pd.to_timedelta(np.arange(12), 'D')

index= pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                        '2015-07-04', '2015-08-04'])
    
data = pd.Series([0, 1, 2, 3], index=index)    
data
data['2015']

pd.date_range('2015-07-03', '2015-07-12')

pd.date_range('2015-07-03', periods=8)

pd.date_range('2015-07-03', periods=8, freq='H')

pd.timedelta_range(0, periods= 9, freq="2H30T")

from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())





df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3)))
                           for i in range(5))

result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')

np.allclose(result1, result2)


result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)



result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)



result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)



__author__ = 'Jie'
import numpy as np
import pandas as pd
from datetime import datetime

# some basic knowledge about the pandas datetime
# e.g.1:  set a datetime index, and remove the original column.
sz000802 = [
    ['2018-07-04', 13.33],
    ['2018-07-05', 14.52],
    ['2018-07-06', 15.97],
    ['2018-07-09', 15.20],
]
df2=pd.DataFrame(sz000802,columns=['date','price'])
df2['date']=pd.to_datetime(df2['date']) # to datetime format
df2.index=pd.DatetimeIndex(df2['date'])
df2=df2.drop(['date'],axis=1)
print (df2)


###  datetime locate, slice, etc
dates=pd.date_range(start='2020-04-01',periods=20)
ts=pd.DataFrame(np.random.randn(20),index=dates)

# index
stamp=ts.index[1]
ts.loc[stamp]
ts.loc['2020-04-01']
# ts.loc['4/1/2020']
# ts.loc['20200401']
# ts.loc['04/01/2020']

# slice by a series of data
ts.loc['2020/04'] #  select all data from 2020-04
ts.loc['2020'] # all data from 2020


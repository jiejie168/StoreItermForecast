__author__ = 'Jie'
'''
'''
import pandas as pd
import seaborn as snn
import matplotlib.pyplot as plt
'''
df=pd.read_csv("train.csv")
df['date']=pd.to_datetime(df['date'])
df.index=pd.DatetimeIndex(df['date'])
df=df.drop(['date'],axis=1)

# categorize_by_day_of_year=df.index.dayofyear //7
# categorize_by_week_of_year=df.index.weekofyear

categorize_by_week_of_year=lambda df:df.index.dayofyear//7 # create a function as the name of categorize_by_week_of_year
additive={"op":lambda a,b:a+b,"inv":lambda a,b: a-b}
print (df.shape)
print(df.head())

def compute_seasonality(series,categorization):
    df=pd.DataFrame()
    df['values']=series
    df.index=series.index
    df['cat']=categorization(df)
    return df.groupby(by='cat')['values'].mean()

def alter_series_by_season(series,categorization, seasonality,op):
    df=pd.DataFrame()
    df['values']=series
    df.index=series.index
    df['cat']=categorization(df) # e.g., the week numbers for groupby.
    # map(): apply the lambda function to Series.
    #op(): lambda function above
    return op(df['values'],df['cat']).map(seasonality)

sea=compute_seasonality(df['sales'],categorize_by_week_of_year)
forecast=alter_series_by_season(df['sales'],categorize_by_week_of_year, sea,additive['op'])

sea.plot()
plt.show()

mean1=compute_seasonality(df['sales'],categorize_by_week_of_year)
print(mean1.shape)
print (mean1.head())

print (forecast.shape)
print (forecast.head())
'''

dict={'a':1,'b':2}
print (dict)
print (dict['a'])
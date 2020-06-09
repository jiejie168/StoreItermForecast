__author__ = 'Jie'

import numpy as np
import pandas as pd
import datetime

#illustrate functionality on a 2d array
# y=np.array([5,10,5,15,20,25,30,35,5,10,15,20]).reshape(2,-1)
# print (running_view(y, 2))

#from https://stackoverflow.com/a/21230438
# explanation about the use of np.lib.index_tricks.as_strided()
# https://zhuanlan.zhihu.com/p/64933417

def running_view(arr, window, axis=-1):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window

    #numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
    #shape: transform shape of matrix. sequence of int.  (2,5)-> [2,5]->[2,5,2]; 2 : the general row is 2; 5: the specific row in
    # each inner matrix, 2: the specific column in each inner matrix
    # strides: The strides of the new array. Defaults to x.strides. sequence of ints.
    """
    shape=arr.shape
    shape1 = list(arr.shape)
    shape1[axis] -= (window-1)  # [2,5]->[2,5,window-1]
    assert(shape1[axis]>0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape1 + [window],
        arr.strides + (arr.strides[axis],))

def add_dates_to_array(array,start_date):
    '''
    :param array:
        array: array-like (np.array,pd.Series,etc.)
    :param start_date: date
        date corresponding to the first value of the series
    :return:
    pandas series containing the same values as the array with a  daily
    DatetimeIndex starting at start_date.
    '''
    s=pd.Series(array)
    dates=pd.date_range(start=start_date,periods=len(array))
    s.index=pd.DatetimeIndex(dates) # dates as the index of series
    return s

class Rolling ():
    """
    a help function for training and forecasting single-variable time series
    window: int
        specify how many days from the past to make available for future computation
        warning : big window may slow-down computation and increase the amount of data
        for the utility to work - periods shorter than window are not currently supported.
        default:365 days
    extract_features:  (np.array,date,object)-> np.array
        this function that extracts features from previous values

     pretransform: (np.array,date)-> np.array
        a function called on input data, used to convert it to a supported format

     posttransform: (np.array,date)-> np.array
        a function called on generated data, used to undo any conversions for pretransform

    """

    def __init__(self,window=365,
                 extract_features=lambda pre_values,current_date,metadata:pre_values,
                 pretransform=lambda values, start_date: values,
                 posttransform= lambda values, start_date: values ):
        self.window=window
        self.extract_features=extract_features
        self.vexextract_features=np.vectorize(extract_features)
        self.pretransform=pretransform
        self.posttransform=posttransform

    def make_training_data(self,value_series,metadata=None,start_date=None):
        """
        generates training data from a raw series of values
        :param value_series: np.array or pd.Series
             values of some time series for forecasting
        :param metadata: object
            any data corresponding to the first value in value_series

        :param start_date:date or None
             the date corresponding to the first value in value_series
             if not set, value_series has to  be a pandas series with a DatetimeIndex
             with 1 day interval

        :return: (np.array, np.array)
            tuple of arrays X and y which are training data in the format supported by lib like sklearn
        """
        if start_date is None:
            start_date=value_series.index.min()
            print (start_date)
        value_series=self.pretransform(value_series,start_date)
        # print (value_series)
        X_base=value_series[:-1] # remove the last row of data
        print (X_base.shape)
        # print (X_base)
        X=[]
        for i, row in enumerate(running_view(X_base,self.window)):
            X.append(self.extract_features(row,start_date+datetime.timedelta(days=i)),metadata)
        X=np.array(X)
        y=value_series[self.window:]
        assert len(X) == len(y)
        return X,y




    def predict(self):
        pass




df = pd.read_csv('train.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = pd.DatetimeIndex(df['date'])
df=df.drop(['date'],axis=1)
# print (df.head())
train = df.iloc[df.index < '2017-01-01']
test = df.iloc[df.index >= '2017-01-01'].iloc[:90]

r = Rolling(window=365)
train_X, train_y = r.make_training_data(train)




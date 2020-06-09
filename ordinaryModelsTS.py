"""
this code includes some typical models for the time series problems
Store Item Demand Forecasting Challenge
Predict 3 months of item sales at different stores

Overview:
This competition is provided as a way to explore different time series techniques on a relatively simple and clean dataset.

You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

What's the best way to deal with seasonality? Should stores be modeled separately, or can you pool them together?
Does deep learning work better than ARIMA? Can either beat xgboost?

This is a great competition to explore different models and improve your skills in forecasting.

"""
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
import statsmodels.tsa.api as smt
import datetime as dt
import itertools

#########################################################################
# load the train data.
df=pd.read_csv("train.csv")
dt=pd.to_datetime(df.date,format="%Y-%m-%d")
df.index=pd.DatetimeIndex(dt)
df=df.drop(['date'],axis=1)
df.head()
#########################################################################

#########################################################################
# some useful functions for the following predictions.
# # Holt-Winters and SARIMA Models Utils
# # Seasonality model

# dictionary for the further selection of seasonality mode
additive={"op":lambda a,b:a+b,"inv":lambda a,b: a-b}
multiplicative={"op":lambda a,b:a*b,"inv":lambda a,b:a/b}

#choose seasonality model from the created dictionary
seasonality_model=additive
#seasonality_model=multiplicative
# create an anonymous function as the name of "categorize_by_week_of_year".
#  how to use:  categorize_by_week_of_year(df). return : e.g.: df.index.dayofweek
categorize_by_week_of_year=lambda df:df.index.dayofyear//7
categorize_by_day_of_week = lambda df:df.index.dayofweek

def compute_seasonality(series,categorization):
    """
    compute seasonal component parameters based on provided series
    
    :type  series: pd.series
    :param categorization: Function used to split values into various 
    period of season
    :type categorization: pd.DataFrame-> some categorized type. e.g. int.
    # e.g. categorize_by_week_of_year(df)
    return: the mean value of each categorise to get the seasonal variation
    """
    df=pd.DataFrame()
    df['values']=series
    df.index=series.index
    df['cat']=categorization(df) # e.g.,categorization(df)-> Int64Index([0,0,0,...1,1,..52],dtype='int64',name='date',length=913000)
    return df.groupby(by='cat')['values'].mean()

def alter_series_by_season(series,categorization, seasonality,op):
    """
    op: the specific seasonality model. e.g.,seasonality_model['inv']
    seasonality: object from the function of compute_seasonality(series,categorization)
    return:
    """
    df=pd.DataFrame()
    df['values']=series
    df.index=series.index
    df['cat']=categorization(df) # e.g., the week numbers for groupby.
    # map(): apply the lambda function to Series.
    # op(): lambda function as showed above.
    return op(df['values'],df['cat']).map(seasonality)

def add_seasonal_component(series,categorization,seasonality):
    """
    add previously computed seasonal component back into a deseasonalized series.
    
    :type series: pd.Series
    :param categorization: Function used to split values into various periods of the season.
    :type categorization: pd.DataFrame -> some categorical type, eg. int
    :param seasonality: value returned from compute_seasonality
    :returns: Series with added seasonal component.
    :rtype: pd.Series
    """
    return alter_series_by_season(series,categorization,seasonality,
                                  seasonality_model["inv"])

def remove_seasonal_component(series, categorization, seasonality):
    """
    Try removing a previously computed seasonal component.
    
    :type series: pd.Series
    :param categorization: Function used to split values into various periods of the season.
    :type categorization: pd.DataFrame -> some categorical type, eg. int
    :param seasonality: value returned from compute_seasonality
    :returns: Deseasonalized series.
    :rtype: pd.Series
    """
    return alter_series_by_season(series, categorization, seasonality, seasonality_model["inv"])

def train_and_forecast(data,categorization,trainer,forecaster,deseasonize,steps_to_forecast=90):
    """
    used for dataset with an item in a single specific store
    this function is used to split input data, deseasonalizes train data, 
    training use the provided trainer
    finally, forecast using forecaster.
    
    :param data: dataset with the training data
    :param categorization: Function used to split values into various periods of the season.
    :type categorization: pd.DataFrame -> some categorical type, e.g. int
    :param trainer: Function used to train the model
    :type trainer: pd.DataFrame -> model
    :param forecaster: (model, steps) -> prediction
    :param steps_to_forecast:  number of steps to forecast
    :return:  a dataframe with:
                                date
                                sales - true values
                                forecast - forecasted values
    """
    # prepare training and validation dataset
    df_train=data.iloc[:-365].copy()  # date for training
    df_validation=data.iloc[-365:].copy() # data in terms of the last 365 rows for validation
    df_train.index=pd.DatetimeIndex(df_train['date'])
    df_validation.index=pd.DatetimeIndex(df_validation['date']) # set datetime as index
    
    if deseasonize:
        # true or Fasle
        seas=compute_seasonality(df_train['sales'],categorization) # mean value of each categorize
        series=remove_seasonal_component(df_train["sales"], categorization, seas)
        df_train['sales']=series
    df_train=df_train.reset_index(drop=True)
    
    # train
    model=trainer(df_train) # train the time series
    
    # forecast
    forecast=forecaster(model,steps_to_forecast)
    
    # create pandas series based on the forecast resutls
    forecast=pd.Series(forecast)
    forecast.name="sales"
    forecast.index=pd.DatetimeIndex(start='2017-01-01',
                                   freq='D',
                                   periods=forecast.size)
    
    if deseasonize:
        forecast=add_seasonal_component (forecast,categorization,seas)
    final_forecast=pd.DataFrame()
    final_forecast['real_values']=df_validation['sales'][:steps_to_forecast]
    final_forecast['forecast']=forecast
    
    return final_forecast

def extract_single_df(data,store,item):
    """
    extract single store/item time series from provided dataset
    
    :param data: Pandas dataframe with multiple timeseries
    :param store: number of the store
    :param item: number of the item
    :return: Pandas dataframe with single store/item time series
    """
    df_single=data.loc[(data.store==store)&(data.item==item),['date','sales']].copy()  # boolen mask 
    df_single.reset_index(drop=True,inplace=True)
    df_single.date=pd.to_datetime(df_single.date)
    return df_single

def smape(y,y_pred):
    """
    compute the SMAPE metrics
    :param y: array with true values
    : param y_pred: array with forecasted values
    :returns: average smape metrics for the given periods
    """
    div=(abs(y_pred)+abs(y))/2
    errors=abs(y_pred-y)/div

    smape=np.sum(errors)/len(y)  # the summation of all error/len(y)
    return smape

def compute_avg_smape(df_y,df_y_pred):
    """
    Compute average SMAPE of multiple forecast

    :param df_y: data series with real values
    :param df_y_pred: dataframe with multiple forecasts
    :returns: average SMAPE of all forecasts
    """
    avg_smape=0
     # dy_y_pred.shape:(a,b), select every single column
    for i in range(df_y_pred.shape[1]):
        err=smape(y=df_y.iloc[:,i],
                 y_pred=df_y_pred.iloc[:,i])
        avg_smape +=err
    avg_smape /=df_y_pred.shape[1]
    return avg_smape

def compute_all_models(data,ids,categorization,trainer,forecaster, deseasonize, steps_to_forecast=90):
    """
    Train the models and use them to make forecast for all of the individual
    time series separately
    
    :params data: dataframe with multiple time series
    :params ids: list of tuples with stores and items
    :param categorization: Function used to split values into various periods of the season.
    :type categorization: pd.DataFrame -> some categorical type, eg. int
    :param trainer: Function used to train the model
    :type trainer: pd.DataFrame -> model
    :param forecaster: (model, steps) -> prediction
    :param steps_to_forecast: number of steps to forecast
    """
    all_models_forecast={} # a dictionary
    all_models_smape=np.array([])
    number_of_time_series=0
    for store,item in ids:
        single_time_series=extract_single_df(data,store,item) # series with a single item in a specific store
        # final_forecast: dataframe: with "real_values" and "forecast" columns only for the validated data
        predictions=train_and_forecast(single_time_series,categorization,
                                         trainer, forecaster, deseasonize, steps_to_forecast)
        score=smape(predictions['real_values'],predictions['forecast'])
        results={
            "item":item,
            "store":store,
            "predictions":predictions,
            "smape":score
        }
        print (score)
        all_models_smape=np.append(all_models_smape,score)
        all_models_forecast[str(store)+str(item)]=results # add every results into the dictionary
        print ("current predicting id of store and item: {}".format(str(store)+str(item)))
        number_of_time_series +=1
    forecast_smape=np.sum(all_models_smape)/number_of_time_series
        
    return all_models_forecast,forecast_smape

################################################################################################
# # Data import and preparation
store_item_data=pd.read_csv("train.csv")
store_item_data.index=pd.DatetimeIndex(store_item_data['date'])

stores=store_item_data['store'].unique()
items=store_item_data['item'].unique()

number_of_stores=10
number_of_items=50

################################################################################################
# create every combination of the store and item.
ids= list(itertools.product(range(1,number_of_stores+1), range(1,number_of_items+1)))
print (ids[:5])
print (len(ids))

# an alternative method to create the list ids. 
# temp=np.ones((50,))
# ids=[]
# for i in range(1,len(stores)+1):
#     store_i=i*temp.astype("int32")
#     id_temp=list(zip(store_i,items))
#     ids.extend(id_temp)
# print (ids[:5])

################################################################################################
# #  Holt-Winters Method
# create an anonymous function which can be used to train all the models: df is the variable here
# the slice indicate the set for validate.
hw_trainer=lambda df:smt.ExponentialSmoothing(df.loc[-365:,'sales'],damped=False,seasonal="add",
                                              trend='add',seasonal_periods=7).fit()
# create a forecaster function to get the forecasts for all the models
hw_forecaster=lambda model, steps: model.predict(steps)

hw_results,hw_smape=compute_all_models(store_item_data, ids, categorize_by_week_of_year,
                                         hw_trainer, hw_forecaster, False)
print("Holt Winters method SMAPE on the validation set: ", hw_smape)
print (hw_results)
print (hw_results["110"])
################################################################################################

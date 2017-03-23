import os
import pandas as pd
import numpy as np
import json
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from zipline.utils.factory import load_bars_from_yahoo
import pytz
import talib
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import ystockquote

dates = pd.date_range('2011-10-13', '2016-10-12')

def get_stocks_info(symbol):
    dict_stocks = ystockquote.get_historical_prices(symbol, '2011-10-13', '2016-12-12')
    json_str = json.dumps(dict_stocks)
    df =pd.read_json(json_str,orient='index')
    df = df.reindex(df.index.rename('Date'))
    return df

def fill_missing(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

ticker='GOOGL'

def getDf(ticker='GOOGL'):
	df = get_stocks_info(ticker)
	new_df = df.filter(['Date','Adj Close'], axis=1)
	adj_df = new_df.reindex(dates, fill_value=np.nan)
	fill_missing(adj_df)
	adj_df = adj_df.reindex(adj_df.index.rename('Date'))
	#adj_df.head()
	return df,adj_df

def smavg(df, N):
    return pd.rolling_mean(df, N)[N - 1:].rename(columns={'Adj Close':'sma'+str(N)})

def expmavg(df, span):
    return pd.ewma(df, span=span).rename(columns={'Adj Close':'ema10'})

def com_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    return daily_returns.rename(columns={'Adj Close':'drr'})

def com_momentum(df, n):
    mom = (df / df.shift(n))
    return mom.rename(columns={'Adj Close':'mom'})

def com_BBW(data,length):
    return 4*pd.stats.moments.rolling_std(data,length).rename(columns={'Adj Close':'bbw'})

def com_williamspercent(df):
    high=pd.rolling_max(df,2)
    low=pd.rolling_min(df,2)
    return (((high-df)/(high-low))*100).rename(columns={'Adj Close':'will'})

def com_rsi(df, n):
    df = df.diff()
    df = df[1:]
    up, down = df.copy(), df.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = pd.rolling_mean(up, n)
    roll_down = pd.rolling_mean(down.abs(), n)
    rs = roll_up / roll_down
    rss = 100.0 - (100.0 / (1.0 + rs))
    return rss.rename(columns={'Adj Close':'rsi'})

def getXY(ticker='GOOGL',time_period=30):
	df,adj_df =getDf(ticker)
	temp_df = adj_df.copy()
	sum_df = df.copy()
	del sum_df['Adj Close']
	sum_df=temp_df.join(sum_df)
	fill_missing(sum_df)
	# print len(sum_df)
	# print len(temp_df)
	low = sum_df['Low']
	high =sum_df['High']
	volume = sum_df['Volume'].astype(float)
	close = sum_df['Close']
	SMA = smavg(temp_df, time_period)
	EMA = expmavg(temp_df,time_period)
	MOM = com_momentum(temp_df, time_period)
	DRR = com_daily_returns(temp_df)
	BBW = com_BBW(temp_df, time_period)
	WILL = com_williamspercent(temp_df)
	RSI = com_rsi(temp_df, time_period)
	ATR = talib.ATR(high.values, low.values,close.values, time_period)
	ROCP = talib.ROCP(temp_df.values.flatten(), time_period)
	OBV = talib.OBV(temp_df.values.flatten(), volume.values)
	ROCR = talib.ROCR(temp_df.values.flatten(), time_period)
	MEDPRICE = talib.MEDPRICE(high.values, low.values)
	MFI = talib.MFI(high.values,low.values,close.values, volume.values, time_period)
	TA_lib_df = pd.DataFrame({
	        'ATR': ATR,
	        'MEDPRICE': MEDPRICE,
	        'MFI': MFI,
	        'ROCP':ROCP,
	        'ROCR': ROCR,
	    }, index=temp_df.index)
	TA_lib_df=TA_lib_df.reindex(dates, fill_value=np.nan)
	X=pd.concat([EMA,SMA,RSI,DRR,MOM,WILL,volume,high,close,TA_lib_df], axis=1)
	fill_missing(X)
	col_list = [s for s in list(X.columns) if 'Volume' in s]
	X[col_list] = X[col_list].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
	Y=pd.concat([X.copy(),temp_df.copy()], axis=1,join_axes=[X.index]).dropna()
	index_values = X.index.values
	Y = temp_df.reindex(index_values, fill_value=np.nan)
	return X,Y
X,Y = getXY(ticker)
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:42:57 2022

@author: ASUS
"""

import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import random


data = pd.read_csv('4g.csv')
data = data[['Date', 'IPTHP_DL']]

data = (data
 .assign(Day=lambda d: pd.to_datetime(d['Date']))
 .set_index('Day')
 .drop(columns='Date')
 )
data = data.asfreq(pd.infer_freq(data.index))

plt.figure(figsize=(10,4))
plt.plot(data )

plt.figure(figsize=(10,4))
plt.plot(data)
#for year in range(2021,2021):
#    plt.axvline(datetime(year,11,20), color='k', linestyle='--', alpha=0.5)
    
stl = STL(data)
result = stl.fit()

seasonal, trend, resid = result.seasonal, result.trend, result.resid

plt.figure(figsize=(8,6))

plt.subplot(4,1,1)
plt.plot(data)
plt.title('Original Series', fontsize=16)

plt.subplot(4,1,2)
plt.plot(trend)
plt.title('Trend', fontsize=16)

plt.subplot(4,1,3)
plt.plot(seasonal)
plt.title('Seasonal', fontsize=16)

plt.subplot(4,1,4)
plt.plot(resid)
plt.title('Residual', fontsize=16)

plt.tight_layout()


estimated = trend + seasonal
plt.figure(figsize=(12,4))
plt.plot(data)
plt.plot(estimated)


resid_mu = data.IPTHP_DL.mean()
resid_dev = data.IPTHP_DL.std()

lower = resid_mu - 0.5*resid_dev
upper = resid_mu + 0.3*resid_dev


plt.figure(figsize=(10,4))
plt.plot(resid)

#plt.fill_between([datetime(2003,11,20), datetime(2015,11,10)], lower, upper, color='g', alpha=0.25, linestyle='--', linewidth=2)
#plt.xlim(datetime(2013,11,20), datetime(2015,12,1))

anomalies = data[(data.IPTHP_DL < lower) ]

plt.figure(figsize=(10,4))
plt.plot(data)
#for year in range(2013,2015):
#    plt.axvline(datetime(year,11,20), color='k', linestyle='--', alpha=0.5)
    
plt.scatter(anomalies.index, anomalies.IPTHP_DL, color='r', marker='D')
print('anomaly: ')
print(anomalies)

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:44:37 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

data = pd.read_csv('webTraffic.csv')
data = (data
 .assign(Day=lambda d: pd.to_datetime(d['Date']))
 .set_index('Day')
 .drop(columns='Date')
 )

data = data.asfreq(pd.infer_freq(data.index))



model = IsolationForest(contamination=0.004)
model.fit(data[['Visite']])

data['outliers']=pd.Series(model.predict(data[['Visite']])).apply(lambda x: 'yes' if (x==-1) else 'no')

data.query('outliers == "yes"')

print(data)




























#plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(15, 6))
plt.plot(data.Date,data.IPTHP_DL)

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


resid_mu = resid.mean()
resid_dev = resid.std()

lower = resid_mu - 3*resid_dev
upper = resid_mu + 3*resid_dev

plt.figure(figsize=(10,4))
plt.plot(resid)

#plt.fill_between([datetime(2003,11,20), datetime(2015,11,10)], lower, upper, color='g', alpha=0.25, linestyle='--', linewidth=2)
#plt.xlim(datetime(2021,6,6), datetime(2021,5,8))

anomalies = data[(resid < lower) | (resid > upper)]

plt.figure(figsize=(10,4))
plt.plot(data.Date,data.IPTHP_DL)
#for year in range(2021,2021):
#    plt.axvline(datetime(year,6,6), color='k', linestyle='--', alpha=0.5)
    
plt.scatter(anomalies.Date, anomalies.IPTHP_DL, color='r', marker='D')
#
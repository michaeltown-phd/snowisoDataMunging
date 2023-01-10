#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:42:05 2022

This script exports snowiso data peaks into a csv file for manual editing

Resamples gcnet temperature data into monthly time series, finds peaks, 
and then saves into csv file

@author: michaeltown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os as os


fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);

# writes entire data file to csv
# df_iso.to_csv(fileLoc+fileNameIso[:-3]+'csv')

# writes peaks file to csv
cols = ['d18O','depthAcc_reg']
# just look at coreID = 1, year = 2018 to start

cid = np.arange(1,6)
years = np.arange(2017,2020)
for c in cid:
    for y in years:
        df_isoPeaks = df_iso[(df_iso.year == y)&(df_iso.coreID ==c)&(df_iso.peaks==1)]
        # add all the lowest values from each date (the top of the snowpack value)
    
        dfTemp = df_iso[(df_iso.year == y)&(df_iso.coreID ==c)&(df_iso.sampleDepthNum == 1)]
        df_isoPeaks = df_isoPeaks.append(dfTemp)        
        df_isoPeaks = df_isoPeaks.sort_index()
        df_isoPeaks[cols].to_csv(fileLoc+'eastGRIPpeak'+str(y)+'p'+str(c)+'.csv')


# write csv for temperature age horizons from gc-net data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
listDir = os.listdir(fileLoc)

ld = [l for l in listDir if 'eastGRIPdepthAge20' in l]

# cleans the '-'s from these files, makes everything on the end of the month
for l in ld:
    df = pd.read_csv(fileLoc+l)
    df.snowDate = df.snowDate.apply(str)
    df.snowDate = df.snowDate.str.replace('-','').str.replace('0715','0731')
    df.drop(columns = ['Unnamed: 0'],inplace = True)
    df.to_csv(fileLoc+l)


# load the met data from gc-net
fileLoc2 = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/GC-Net/'
df_gcnet = pd.read_csv(fileLoc2+'eastGRIPgcnet2014-2019sfcMet.csv')
# convert year and julian date into datetime variable
df_gcnet['datetime'] = pd.to_datetime(df_gcnet.year,format = '%Y')+pd.to_timedelta(df_gcnet.julianDate, unit = 'd')
df_gcnet.set_index('datetime',inplace = True)
# plot the temperature data as gut check
fig = plt.figure()
plt.plot(df_gcnet.index,df_gcnet.Tair1,color = 'blue',alpha = 0.5,label = 'Tair1')
plt.plot(df_gcnet.index,df_gcnet.Tair1,color = 'red',alpha = 0.5,label = 'Tair2')
plt.ylabel('T (oC)')
plt.xlabel('date')
plt.legend()
fig = plt.figure()
plt.plot(df_gcnet.index,df_gcnet.Tair1-df_gcnet.Tair2,color = 'black',alpha = 0.5,label = 'resid T1-T2')
plt.ylabel('resid T (oC)')
plt.xlabel('date')
plt.legend()
np.mean(df_gcnet.Tair1-df_gcnet.Tair2)

# resample the temperature time series to a monthly time series then
df_gcnetMonthly = df_gcnet.resample('M').mean()
df_gcnetWeekly = df_gcnet.resample('W').mean()
df_gcnetDaily = df_gcnet.resample('D').mean()
fig = plt.figure()
plt.plot(df_gcnet.index,df_gcnet.Tair1,color = 'black',alpha = 0.2,label = 'hour')
plt.plot(df_gcnetDaily.index,df_gcnetDaily.Tair1,color = 'black',alpha = 0.5,label = 'day')
plt.plot(df_gcnetWeekly.index,df_gcnetWeekly.Tair1,color = 'black',alpha = 0.7,label = 'day')
plt.plot(df_gcnetMonthly.index,df_gcnetMonthly.Tair1,color = 'gray',alpha = 1,label = 'month')
plt.ylabel('T (oC)')
plt.xlabel('date')
plt.title('gcnet 2m air T1')
plt.legend(loc = 'lower left')
fig.savefig('/home/michaeltown/work/projects/snowiso/figures/EastGRIP/meteo/eastGRIP2mAirTempGcnet2014-2019.jpg')


# find peaks in the monthly data set
# peak params
dist = 2; 
wid = 1; 
hei = None;
prom= None;

peaks, _ = find_peaks(df_gcnetMonthly.Tair1, distance = dist, height = hei, width = wid, prominence = prom)
troughs, _ = find_peaks(-df_gcnetMonthly.Tair1, distance = dist, height = hei, width = wid, prominence = prom)
maxMin = np.append(peaks,troughs)

fig = plt.figure()
plt.plot(df_gcnet.Tair1,df_gcnet.index,color = 'black',alpha = 0.2,label = 'hour')
plt.plot(df_gcnetDaily.Tair1,df_gcnetDaily.index,color = 'black',alpha = 0.5,label = 'day')
plt.plot(df_gcnetWeekly.Tair1,df_gcnetWeekly.index,color = 'black',alpha = 0.7,label = 'day')
plt.plot(df_gcnetMonthly.Tair1,df_gcnetMonthly.index,color = 'white',alpha = 0.8,label = 'month')
plt.plot(df_gcnetMonthly.Tair1[maxMin],df_gcnetMonthly.index[maxMin],'x',color = 'orange')

plt.xlabel('T (oC)')
plt.ylabel('date')
plt.title('gcnet 2m air T1')
plt.legend(loc = 'lower left')
fig.savefig('/home/michaeltown/work/projects/snowiso/figures/EastGRIP/meteo/eastGRIP2mAirTempGcnet2014-2019_vert.jpg')

# write the csv file
cols = ['Tair1']
df_gcnetPeaks = pd.DataFrame(df_gcnetMonthly.Tair1[maxMin].sort_index(ascending = False))
np.round(df_gcnetPeaks[cols],2).to_csv(fileLoc2+'eastGRIPgcnetTair1peaks2014-2019.csv')


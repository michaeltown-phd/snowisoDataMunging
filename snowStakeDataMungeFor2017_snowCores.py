#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:51:42 2023

This code will munge the EastGRIPsnowStake.pkl data into a format that facilitates
inclusion into the EastGRIP 2017 snow core data stream.

@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import EastGRIPprocs as eg
import matplotlib.pyplot as plt
import figureMagic as fm

# figure location
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/2017/'

# load snowstake data
fileLocMeteo = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
fileName = 'EastGRIPsnowStakes2016_2019.pkl'
df_ss = pd.read_pickle(fileLocMeteo+fileName)

# load snow core dates
fileLocIso = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileName = 'snowCoreDates2017.csv'
df_dates = pd.read_csv(fileLocIso+fileName)
df_dates.dates = pd.to_datetime(df_dates.dates,infer_datetime_format=True)

# isolate the 2017 data 

beg = np.min(df_dates.dates)
end = np.max(df_dates.dates)
df_ss = df_ss[(df_ss.index >= beg)&(df_ss.index <= end)]

df_ss_int = eg.dfInterpTime(df_ss,'1D','linear')

# pull out the dates that we need from df_ss_int
ixs = df_ss_int.index.intersection(df_dates.dates)
df_ss_int_intersc = df_ss_int.loc[ixs]

# check work here
fig = plt.figure()
plt.plot(df_ss.index,df_ss.stakeHeight,'o',color = 'black',markersize = 4, label = 'daily interp')
plt.plot(df_ss_int.index,df_ss_int.stakeHeight,color = 'blue',linewidth = 6, alpha = 0.2,label = 'original data')
plt.plot(df_ss_int_intersc.index,df_ss_int_intersc.stakeHeight,'^',color = 'orange',markersize = 4,label = 'snow core extraction dates')
plt.xticks(rotation = 25)
plt.ylabel('snow stake height (cm)')
plt.legend()

fm.saveas(fig, figureLoc + 'snowStakeInterp')
df_ss_int_intersc.drop(columns = ['year','dayOfYear'],inplace = True)
df_ss_int_intersc.to_csv(fileLocMeteo + 'snowStakeInterpSnowCore2017.csv')



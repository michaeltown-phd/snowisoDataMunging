#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:07:10 2022

This script will take mean snow core profiles from 2017-2019 and stack them on each other 
according to the accumulation rate, then according to manually assigned dates pulled from 
features in the EastGRIP PROMICE temperature data set.

@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import pickle as pkl
import figureMagic as fm
import calendar
import datetime as datetime

# functions



#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
plt.rcParams['text.latex.preamble']=[r"\usepackage{wasysym}"]
pptsym = 'ppt' # '\textperthousand'



# take a decimal value for the year and convert to the days in a year
# returns the number of days and remainder
def decimalToDaysInYear(ily,d):
    daysInYear = 365 + ily  
    return np.round(d*daysInYear,0), d*daysInYear-np.round(d*daysInYear,0)
    

# read in the mean profiles and associated statistics

fileLocIso = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileLocMeteo = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'

dataFilename2019 = 'eastGRIP_SCmeanProfileData_2019.pkl'
dataFilename2018 = 'eastGRIP_SCmeanProfileData_2018.pkl'
dataFilename2017 = 'eastGRIP_SCmeanProfileData_2017.pkl'
dataFilenameAcc = 'Sonic_mean.txt'

df_2019 = pd.read_pickle(fileLocIso+dataFilename2019);
df_2018 = pd.read_pickle(fileLocIso+dataFilename2018);
df_2017 = pd.read_pickle(fileLocIso+dataFilename2017);

df_acc = pd.read_csv(fileLocMeteo+dataFilenameAcc,sep='\t')

# plot the accumulation data
plt.figure()
plt.plot(df_acc.Julian_Sonic_mean,df_acc.Sonic_cumsum_snow_height)

# convert julian date into datetime variable

df_acc['year'] = np.floor(df_acc.Julian_Sonic_mean)
df_acc['julDay'],df_acc['decDay'] = decimalToDaysInYear(df_acc.year.astype(int).apply(calendar.isleap), df_acc.Julian_Sonic_mean-df_acc.year)

df_acc['date'] = df_acc.year.astype(int).astype(str)+'-'+df_acc.julDay.astype(int).astype(str)
df_acc['datetime'] =pd.to_datetime(df_acc.date,format = '%Y-%j')
plt.figure()
plt.plot(df_acc.datetime,df_acc.Sonic_cumsum_snow_height)

jul2019 = datetime.datetime.strptime('20190716', '%Y%m%d')
jul2018 = datetime.datetime.strptime('20180716', '%Y%m%d')
jul2017 = datetime.datetime.strptime('20170716', '%Y%m%d')
acc2017 = df_acc[df_acc.datetime == jul2018].Sonic_cumsum_snow_height.values-df_acc[df_acc.datetime == jul2017].Sonic_cumsum_snow_height.values
acc2018 = df_acc[df_acc.datetime == jul2019].Sonic_cumsum_snow_height.values-df_acc[df_acc.datetime == jul2018].Sonic_cumsum_snow_height.values

lbstd = (df_2019.d18O-df_2019.d18O_std)*np.nan;
ubstd = (df_2019.d18O+df_2019.d18O_std)*np.nan;
lbmin = df_2019.d18O_min*np.nan;
ubmax = df_2019.d18O_max*np.nan;
_,_ =fm.myDepthFunc(df_2019.d18O,-df_2019.index,df_2019.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017-2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],None,'prof_d18O_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_2018.d18O,-df_2018.index-acc2018,color = 'black',linewidth = 3,alpha = 0.5,label='2018')
plt.plot(df_2017.d18O,-df_2017.index-acc2018-acc2017,color = 'black',linewidth = 3,alpha = 0.3,label='2017')
plt.ylim(-200,15)
plt.legend(loc='lower left')

lbstd = (df_2019.dexcess-df_2019.dexcess)*np.nan;
ubstd = (df_2019.dexcess+df_2019.dexcess)*np.nan;
lbmin = df_2019.dexcess_min*np.nan;
ubmax = df_2019.dexcess_max*np.nan;
_,_ =fm.myDepthFunc(df_2019.dexcess,-df_2019.index,df_2019.dexcess_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017-2019 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],None,'prof_d18O_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_2018.dexcess,-df_2018.index-acc2018,color = 'blue',linewidth = 3,alpha = 0.5,label='2018')
plt.plot(df_2017.dexcess,-df_2017.index-acc2018-acc2017,color = 'blue',linewidth = 3,alpha = 0.3,label='2017')
plt.ylim(-200,15)
plt.legend(loc='lower left')
'''
lbstd = df_EGRIP_profiles_2019.dD-df_EGRIP_profiles_2019.dD_std;
ubstd = df_EGRIP_profiles_2019.dD+df_EGRIP_profiles_2019.dD_std;
lbmin = df_EGRIP_profiles_2019.dD_min;
ubmax = df_EGRIP_profiles_2019.dD_max;
myDepthFunc(df_EGRIP_profiles_2019.dD,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,2],fileLoc,'prof_dD_EGRIP2019');

lbstd = df_EGRIP_profiles_2019.dexcess-df_EGRIP_profiles_2019.dexcess_std;
ubstd = df_EGRIP_profiles_2019.dexcess+df_EGRIP_profiles_2019.dexcess_std;
lbmin = df_EGRIP_profiles_2019.dexcess_min;
ubmax = df_EGRIP_profiles_2019.dexcess_max;
myDepthFunc(df_EGRIP_profiles_2019.dexcess,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],fileLoc,'prof_dexcess_EGRIP2019');

lbstd = df_EGRIP_profiles_2019.dxsln-df_EGRIP_profiles_2019.dxsln_std;
ubstd = df_EGRIP_profiles_2019.dxsln+df_EGRIP_profiles_2019.dxsln_std;
lbmin = df_EGRIP_profiles_2019.dxsln_min;
ubmax = df_EGRIP_profiles_2019.dxsln_max;
myDepthFunc(df_EGRIP_profiles_2019.dxsln,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,2],fileLoc,'prof_dxsln_EGRIP2019');
'''
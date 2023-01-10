#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:07:10 2022

This script will take mean snow core profiles from 2017-2019 and stack them on each other 
according to the accumulation rate, then according to manually assigned dates pulled from 
features in the EastGRIP PROMICE temperature data set.

Also modified for examining individual profiles. The idea here will be to look at profiles from the
beginning and end of a summer field season and see how they have change wrt to each other.


@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import figureMagic as fm
import calendar
import datetime as datetime

# functions

# pulls out data frame for individual profiles
def makeIndDF(c,df_og):
    cols = ['d18O','dexcess','depthAcc_reg']
    df = pd.DataFrame()
    df['d18O'] = df_og[(df_og.date == c[0])&(df_og.coreID == c[1])].d18O
    df['dexcess'] = df_og[(df_og.date == c[0])&(df_og.coreID == c[1])].dexcess
    df['depthAcc_reg'] = df_og[(df_og.date == c[0])&(df_og.coreID == c[1])].depthAcc_reg
    df.sort_values(by = 'depthAcc_reg',inplace = True)
    df.set_index('depthAcc_reg',inplace = True)
    return df


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
dataFilenameAcc = 'Sonic_mean2016-2019_HCSL.txt'
dataFilenameOG = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'

df_2019 = pd.read_pickle(fileLocIso+dataFilename2019);
df_2018 = pd.read_pickle(fileLocIso+dataFilename2018);
df_2017 = pd.read_pickle(fileLocIso+dataFilename2017);
df_og = pd.read_pickle(fileLocIso+dataFilenameOG)

# load individual profiles from 2017-2019
core1 = [pd.to_datetime('20170526',format = '%Y%m%d'),1]
core2 = [pd.to_datetime('20170725',format = '%Y%m%d'),1]
core3 = [pd.to_datetime('20180527',format = '%Y%m%d'),1]
core4 = [pd.to_datetime('20180721',format = '%Y%m%d'),1]
core5 = [pd.to_datetime('20190529',format = '%Y%m%d'),1]
core6 = [pd.to_datetime('20190724',format = '%Y%m%d'),1]

df201705 = makeIndDF(core1, df_og)
df201707 = makeIndDF(core2, df_og)
df201805 = makeIndDF(core3, df_og)
df201807 = makeIndDF(core4, df_og)
df201905 = makeIndDF(core5, df_og)
df201907 = makeIndDF(core6, df_og)



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
_,_ =fm.myDepthFunc(df_2019.d18O,-df_2019.index,df_2019.d18O_num,'grey',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017-2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],None,'prof_d18O_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_2018.d18O,-df_2018.index-acc2018,color = 'black',linewidth = 3,alpha = 0.3,label='2018')
plt.plot(df_2017.d18O,-df_2017.index-acc2018-acc2017,color = 'black',linewidth = 3,alpha = 0.2,label='2017')
plt.ylim(-200,15)
plt.xlim(-50,-20)
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

# experimental plots with individual profiles

lbstd = (df_2019.d18O-df_2019.d18O_std)*np.nan;
ubstd = (df_2019.d18O+df_2019.d18O_std)*np.nan;
lbmin = df_2019.d18O_min*np.nan;
ubmax = df_2019.d18O_max*np.nan;
fig1, ax, filename =fm.myDepthFunc3(df_2019.d18O,-df_2019.index,df_2019.d18O_num,'grey',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017-2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],'2019',None,'prof_d18O_EGRIP2019_test');
ax.plot(df_2018.d18O,-df_2018.index-acc2018,color = 'black',linewidth = 3,alpha = 0.3,label='2018')
ax.plot(df_2017.d18O,-df_2017.index-acc2018-acc2017,color = 'black',linewidth = 3,alpha = 0.2,label='2017')
ax.fill_betweenx(-df_2019.index,df_2019.d18O_min,df_2019.d18O_max,color = 'black',alpha = 0.1)
ax.fill_betweenx(-df_2018.index-acc2018,df_2018.d18O_min,df_2018.d18O_max,color = 'black',alpha = 0.1)
ax.fill_betweenx(-df_2017.index-acc2018-acc2017,df_2017.d18O_min,df_2017.d18O_max,color = 'black',alpha = 0.1)

# ax.plot(df201705.d18O,-df201705.index-acc2018-acc2017,color = 'black',linewidth = 1,alpha = 0.7)
# ax.plot(df201707.d18O,-df201707.index-acc2018-acc2017,color = 'black',linewidth = 1,alpha = 0.7)
# ax.plot(df201805.d18O,-df201805.index-acc2018,color = 'black',linewidth = 1,alpha = 0.8)
# ax.plot(df201807.d18O,-df201807.index-acc2018,color = 'black',linewidth = 1,alpha = 0.8)
# ax.plot(df201905.d18O,-df201905.index,color = 'black',linewidth = 1,alpha = 0.9)
# ax.plot(df201907.d18O,-df201907.index,color = 'black',linewidth = 1,alpha = 0.9)
plt.ylim(-200,15)
plt.xlim(-50,-20)
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

# representation of monthly air temperature and monthly accumulation vs time, and temp vs acc

fig, ax = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3,1]})
fig.tight_layout()
ax2 = ax[0].twinx()
ax[0].plot(dfJoin.index,dfJoin.AirTemperatureC,color = 'red',linewidth = 3, alpha = 0.5,label = 'Air Temperature')
ax2.plot(dfJoin.index,dfJoin.Sonic_cumsum_snow_height,color = 'gray',linewidth = 5, alpha = 0.5,label = 'Accumulation')
dfAxis = pd.DataFrame(dfJoin.index[0:41:4].date).append(pd.DataFrame([datetime.date(2019,9,1)]))
ax[0].set_xticklabels(dfAxis[0],rotation = 30,ha = 'right')
ax[0].set_ylabel('Temperature (oC)'); ax2.set_ylabel('accumulation (cm)')
ax[1].plot(dfJoin.AirTemperatureC,dfJoin.Sonic_cumsum_snow_height,color = 'red',linewidth = 3, alpha = 0.5)
ax[1].set_xlabel('Temperature (oC)'); ax[1].set_ylabel(None); ax[1].tick_params(axis='y',label1On = False) 
ax2.legend(loc = 'upper left')
ax[0].legend(loc = 'lower right')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:52:24 2022

Figures for the post-dep processes paper, EastGRIP 2016-2019: 
meteorology context

These data have been processed and pickled by eastGRIPpromiceDataMunging.py


@author: michaeltown
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
import pickle as pkl
import figureMagic as fm 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from windrose import WindroseAxes
import matplotlib.cm as cm
import csv

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
os.chdir(fileLoc)
dataFileName = 'eastGRIP_PROMICEData_2016-2019.pkl';

df = pd.read_pickle(dataFileName)

# redo some of the resapling

# resample the data to get smoothed values
df_daily = pd.DataFrame()
df_daily['AirTemperatureC'] = df.AirTemperatureC.resample('D').mean()
df_daily['WindSpeedms'] = df.WindSpeedms.resample('D').mean()
df_daily['SpecificHumiditygkg'] = df.SpecificHumiditygkg.resample('D').mean()
df_daily['ShortwaveRadiationDown_CorWm2'] = df.ShortwaveRadiationDown_CorWm2.resample('D').mean()
df_daily['ShortwaveRadiationUp_CorWm2'] = df.ShortwaveRadiationUp_CorWm2.resample('D').mean()
df_daily['netSolar'] = df_daily.ShortwaveRadiationDown_CorWm2-df_daily.ShortwaveRadiationUp_CorWm2

df_weekly = pd.DataFrame()
df_weekly['AirTemperatureC'] = df.AirTemperatureC.resample('W').mean()
df_weekly['WindSpeedms'] = df.WindSpeedms.resample('W').mean()
df_weekly['SpecificHumiditygkg'] = df.SpecificHumiditygkg.resample('W').mean()
df_weekly['ShortwaveRadiationDown_CorWm2'] = df.ShortwaveRadiationDown_CorWm2.resample('W').mean()
df_weekly['ShortwaveRadiationUp_CorWm2'] = df.ShortwaveRadiationUp_CorWm2.resample('W').mean()
df_weekly['netSolar'] = df_weekly.ShortwaveRadiationDown_CorWm2-df_weekly.ShortwaveRadiationUp_CorWm2


df_monthly = pd.DataFrame()
df_monthly['AirTemperatureC'] = df.AirTemperatureC.resample('M').mean()
df_monthly['WindSpeedms'] = df.WindSpeedms.resample('M').mean()
df_monthly['SpecificHumiditygkg'] = df.SpecificHumiditygkg.resample('M').mean()
df_monthly['ShortwaveRadiationDown_CorWm2'] = df.ShortwaveRadiationDown_CorWm2.resample('M').mean()
df_monthly['ShortwaveRadiationUp_CorWm2'] = df.ShortwaveRadiationUp_CorWm2.resample('M').mean()
df_monthly['netSolar'] = df_monthly.ShortwaveRadiationDown_CorWm2-df_monthly.ShortwaveRadiationUp_CorWm2

plt.rc('font', size = 10)
# plot temperature 2016-2019
date1 = '20160101'
date2 = '20200101'
dateSum1 = pd.to_datetime('20190529', format='%Y%m%d')
dateSum2 = pd.to_datetime('20190724', format='%Y%m%d')

fn = 'eastGRIP_T_' + date1 + '_' + date2 + '_pub'
fig1, ax1 = fm.myTimeSeriesFunc3(df.index, df.AirTemperatureC,'gray', pd.to_datetime(date1, format='%Y%m%d'),
                 pd.to_datetime(date2, format='%Y%m%d'),-80,10,'EastGRIP Temp (PROMICE)','date','T ($^{o}$C)',fileLoc,fn)
ax1.plot(df_daily.index,df_daily.AirTemperatureC,color = 'black',alpha = 0.5)
ax1.plot(df_weekly.index,df_weekly.AirTemperatureC,color = 'black',alpha = 0.7)
ax1.plot(df_monthly.index,df_monthly.AirTemperatureC,color = 'white',alpha = 0.7,linewidth = 3)
inset_ax = inset_axes(ax1, width="30%", height='30%', loc=3)
inset_ax.tick_params(axis="y",direction="in", pad=-35)
inset_ax.tick_params(axis="x",direction="in", pad=-15)
inset_ax.hist(df.AirTemperatureC[(df.index>date1)&(df.index<date2)],color = 'gray',density = 1,alpha = 0.5)
inset_ax.patch.set_alpha(0.3)
inset_ax.text(-35, 0.01, 'n = ' + str(df.AirTemperatureC[(df.index>date1)&(df.index<date2)].count()))
inset_ax.text(-50,0.023,'hourly T ($^{o}$C)')
inset_ax.yaxis.set_ticks([0.01,0.02]) 
inset_ax.xaxis.set_ticks([-60,-40,-20,0]) 
inset_ax.set(ylim = [0,0.03])
ax1.legend(['hourly','daily','weekly','monthly'],loc='lower right',facecolor = 'royalblue')
ax1.axvspan(dateSum1, dateSum2,-80,10, alpha=0.5, color='orange')
fig1.savefig(fn+'.jpg')


# plot temperature summer 2019
date1 = '20190501'
date2 = '20190901'
dateSum1 = pd.to_datetime('20190529', format='%Y%m%d')
dateSum2 = pd.to_datetime('20190724', format='%Y%m%d')

fn = 'eastGRIP_T_' + date1 + '_' + date2 + '_pub'
fig1, ax1 = fm.myTimeSeriesFunc3(df.index, df.AirTemperatureC,'gray', pd.to_datetime(date1, format='%Y%m%d'),
                 pd.to_datetime(date2, format='%Y%m%d'),-80,10,'EastGRIP Temp (PROMICE)','date','T ($^{o}$C)',fileLoc,fn)
ax1.plot(df_daily.index,df_daily.AirTemperatureC,color = 'black',alpha = 0.5)
ax1.plot(df_weekly.index,df_weekly.AirTemperatureC,color = 'black',alpha = 0.7)
ax1.plot(df_monthly.index,df_monthly.AirTemperatureC,color = 'white',alpha = 0.7,linewidth = 3)
inset_ax = inset_axes(ax1, width="30%", height='30%', loc=3)
inset_ax.tick_params(axis="y",direction="in", pad=-35)
inset_ax.tick_params(axis="x",direction="in", pad=-15)
inset_ax.hist(df.AirTemperatureC[(df.index>date1)&(df.index<date2)],color = 'gray',density = 1,alpha = 0.5)
inset_ax.patch.set_alpha(0.3)
inset_ax.text(-35, 0.025, 'n = ' + str(df.AirTemperatureC[(df.index>date1)&(df.index<date2)].count()))
inset_ax.text(-50,0.075,'hourly T ($^{o}$C)')
inset_ax.yaxis.set_ticks([0.05]) 
inset_ax.xaxis.set_ticks([-60,-40,-20,0]) 
inset_ax.set(ylim = [0,0.1])
ax1.legend(['hourly','daily','weekly','monthly'],loc='lower right',facecolor = 'royalblue')
ax1.axvspan(dateSum1, dateSum2,-80,10, alpha=0.5, color='orange')
fig1.savefig(fn+'.jpg')

# plot wind speed 2016-2019
date1 = '20160101'
date2 = '20200101'
fn = 'eastGRIP_ws_' + date1 + '_' + date2 + '_pub'
fig1, ax1 = fm.myTimeSeriesFunc3(df.index, df.WindSpeedms,'lightblue', pd.to_datetime(date1, format='%Y%m%d'),
                 pd.to_datetime(date2, format='%Y%m%d'),-5,20,'EastGRIP wind speed (PROMICE)','date','wind speed (m/s)',fileLoc,fn)
ax1.plot(df_daily.index,df_daily.WindSpeedms,color = 'blue',alpha = 0.5)
ax1.plot(df_weekly.index,df_weekly.WindSpeedms,color = 'darkblue',alpha = 0.7)
ax1.plot(df_monthly.index,df_monthly.WindSpeedms,color = 'white',alpha = 0.7,linewidth = 3)
inset_ax = inset_axes(ax1, width="30%", height='30%', loc=3)
inset_ax.tick_params(axis="y",direction="in", pad=-20)
inset_ax.tick_params(axis="x",direction="in", pad=-15)
inset_ax.hist(df.WindSpeedms[(df.index>date1)&(df.index<date2)],color = 'gray',density = 1,alpha = 0.5)
inset_ax.patch.set_alpha(0.3)
inset_ax.text(5, 0.1, 'n = ' + str(df.WindSpeedms[(df.index>date1)&(df.index<date2)].count()))
inset_ax.text(3,0.23,'hourly ws (m/s)')
inset_ax.yaxis.set_ticks([0.1,0.2]) 
inset_ax.xaxis.set_ticks([0,10,20]) 
inset_ax.set(ylim = [0,0.3])
ax1.axvspan(dateSum1, dateSum2,-5,20, alpha=0.5, color='orange')
ax1.legend(['hourly','daily','weekly','monthly'],loc='upper left',facecolor = 'gray')
axWr = WindroseAxes(fig1,[0.65,0.65,0.2,0.2])
fig1.add_axes(axWr)
axWr.bar(df.WindDirectiond[(df.index>date1)&(df.index<date2)], df.WindSpeedms[(df.index>date1)&(df.index<date2)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
axWr.set_yticks(np.arange(0, 12, step=2))
#plt.legend(loc = 'upper right', fontsize = 4)
fig1.savefig(fn+'.jpg')

tempEastGRIP2016_2019 = dict()
windEastGRIP2016_2019 = dict()
monthDict = dict()
monthDict['sum'] = [6,7,8]
monthDict['spr'] = [3,4,5]
monthDict['aut'] = [9,10,11]
monthDict['win'] = [12,1,2]

for k in monthDict.keys():
    dfT = df[(df.index.month == monthDict[k][0])|(df.index.month == monthDict[k][1])|
                 (df.index.month == monthDict[k][2])].AirTemperatureC
    dfWS = df[(df.index.month == monthDict[k][0])|(df.index.month == monthDict[k][1])|
                 (df.index.month == monthDict[k][2])].WindSpeedms
    tempEastGRIP2016_2019[k] = [np.round(np.mean(dfT),3),np.round(np.std(dfT),3)]
    windEastGRIP2016_2019[k] = [np.round(np.mean(dfWS),3),np.round(np.std(dfWS),3)]

''' 
summer 2019 side-bar

k = 'sum'

dfT = df[(df.index.month == monthDict[k][0])|(df.index.month == monthDict[k][1])|
             (df.index.month == monthDict[k][2])].AirTemperatureC
dfWS = df[(df.index.month == monthDict[k][0])|(df.index.month == monthDict[k][1])|
                 (df.index.month == monthDict[k][2])].WindSpeedms

np.mean(dfT[dfT.index.year == 2019])
Out[98]: -11.187436594202914

np.std(dfT[dfT.index.year == 2019])
Out[99]: 5.441387611213928

np.mean(dfWS[dfWS.index.year == 2019])
Out[101]: 4.479356884057976

np.std(dfWS[dfWS.index.year == 2019])
Out[102]: 1.7364356930540303

'''


# write the data into a data file

fn = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/EastGRIPMetData2016-2019.csv'

with open(fn,'w') as w:
    w.write('Temperature (oC) EastGRIP 2016-2019\n')
    w.write('___________________________________\n')
    w.write('         mean    std\n')
    

with open(fn, 'a') as f:  
    w = csv.writer(f)
    for k, v in tempEastGRIP2016_2019.items():
       w.writerow([k, v])

with open(fn,'a') as w:
    w.write('\n\n\nWind Speed (m/s) EastGRIP 2016-2019\n')
    w.write('___________________________________\n')
    w.write('         mean    std\n')

with open(fn,'a') as f:
    w = csv.writer(f)
    for k, v in windEastGRIP2016_2019.items():
       w.writerow([k, v])
    
    

# plot wind speed sum 2019
date1 = '20190501'
date2 = '20190901'
fn = 'eastGRIP_ws_' + date1 + '_' + date2 + '_pub'
fig1, ax1 = fm.myTimeSeriesFunc3(df.index, df.WindSpeedms,'lightblue', pd.to_datetime(date1, format='%Y%m%d'),
                 pd.to_datetime(date2, format='%Y%m%d'),-5,20,'EastGRIP wind speed (PROMICE)','date','wind speed (m/s)',fileLoc,fn)
ax1.plot(df_daily.index,df_daily.WindSpeedms,color = 'blue',alpha = 0.5)
ax1.plot(df_weekly.index,df_weekly.WindSpeedms,color = 'darkblue',alpha = 0.7)
ax1.plot(df_monthly.index,df_monthly.WindSpeedms,color = 'white',alpha = 0.7,linewidth = 3)
inset_ax = inset_axes(ax1, width="30%", height='30%', loc=3)
inset_ax.tick_params(axis="y",direction="in", pad=-20)
inset_ax.tick_params(axis="x",direction="in", pad=-15)
inset_ax.hist(df.WindSpeedms[(df.index>date1)&(df.index<date2)],color = 'gray',density = 1,alpha = 0.5)
inset_ax.patch.set_alpha(0.3)
inset_ax.text(5, 0.1, 'n = ' + str(df.WindSpeedms[(df.index>date1)&(df.index<date2)].count()))
inset_ax.text(3,0.23,'hourly ws (m/s)')
inset_ax.yaxis.set_ticks([0.1,0.2]) 
inset_ax.xaxis.set_ticks([0,10,20]) 
inset_ax.set(ylim = [0,0.3])
ax1.axvspan(dateSum1, dateSum2,-5,20, alpha=0.5, color='orange')
ax1.legend(['hourly','daily','weekly','monthly'],loc='upper left',facecolor = 'gray')
fig1.savefig(fn+'.jpg')
axWr = WindroseAxes(fig1,[0.65,0.65,0.2,0.2])
fig1.add_axes(axWr)
axWr.bar(df.WindDirectiond[(df.index>date1)&(df.index<date2)], df.WindSpeedms[(df.index>date1)&(df.index<date2)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
axWr.set_yticks(np.arange(0, 12, step=2))
#plt.legend(loc = 'upper right', fontsize = 4)
fig1.savefig(fn+'.jpg')

# plot wind rose 2016-2019
plt.rc('font',size=16)
date1 = '20160101'
date2 = '20200101'
fn = 'eastGRIP_wrose_' + date1 + '_' + date2 + '_pub'
ax = WindroseAxes.from_ax()
ax.bar(df.WindDirectiond[(df.index>date1)&(df.index<date2)], df.WindSpeedms[(df.index>date1)&(df.index<date2)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,13000,'EastGRIP 2016-2019')
fig1.savefig(fn+'.jpg')

# plot wind rose sum 2019
date1 = '20190501'
date2 = '20190901'
fn = 'eastGRIP_wrose_' + date1 + '_' + date2 + '_pub'
ax = WindroseAxes.from_ax()
ax.bar(df.WindDirectiond[(df.index>date1)&(df.index<date2)], df.WindSpeedms[(df.index>date1)&(df.index<date2)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,1150,'EastGRIP Summer 2019')
fig1.savefig(fn+'.jpg')

# plot wind rose sum
date1 = '20160101'
date2 = '20210101'
dfTemp = df[(df.index>date1)&(df.index<date2)]
fn = 'eastGRIP_wrose_summer_2016_2021_pub'
ax = WindroseAxes.from_ax()
ax.bar(dfTemp.WindDirectiond[(dfTemp.index.month == 6)|(dfTemp.index.month == 7)|(dfTemp.index.month == 8)], dfTemp.WindSpeedms[(dfTemp.index.month == 6)|(dfTemp.index.month == 7)|(dfTemp.index.month == 8)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,1150,'EastGRIP Summer')
fig1.savefig(fn+'.jpg')

# plot wind rose spring
date1 = '20160101'
date2 = '20210101'
dfTemp = df[(df.index>date1)&(df.index<date2)]
fn = 'eastGRIP_wrose_spring_2016_2021_pub'
ax = WindroseAxes.from_ax()
ax.bar(dfTemp.WindDirectiond[(dfTemp.index.month == 3)|(dfTemp.index.month == 4)|(dfTemp.index.month == 5)], dfTemp.WindSpeedms[(dfTemp.index.month == 3)|(dfTemp.index.month == 4)|(dfTemp.index.month == 5)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,1150,'EastGRIP Spring')
fig1.savefig(fn+'.jpg')

# plot wind rose autumn
date1 = '20160101'
date2 = '20210101'
dfTemp = df[(df.index>date1)&(df.index<date2)]
fn = 'eastGRIP_wrose_autumn_2016_2021_pub'
ax = WindroseAxes.from_ax()
ax.bar(dfTemp.WindDirectiond[(dfTemp.index.month == 9)|(dfTemp.index.month == 10)|(dfTemp.index.month == 11)], dfTemp.WindSpeedms[(dfTemp.index.month == 9)|(dfTemp.index.month == 10)|(dfTemp.index.month == 11)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,1150,'EastGRIP Autumn')
fig1.savefig(fn+'.jpg')

# plot wind rose winter
date1 = '20160101'
date2 = '20210101'
dfTemp = df[(df.index>date1)&(df.index<date2)]
fn = 'eastGRIP_wrose_winter_2016_2021_pub'
ax = WindroseAxes.from_ax()
ax.bar(dfTemp.WindDirectiond[(dfTemp.index.month == 12)|(dfTemp.index.month == 1)|(dfTemp.index.month == 2)], dfTemp.WindSpeedms[(dfTemp.index.month == 12)|(dfTemp.index.month == 1)|(dfTemp.index.month == 2)],  normed = False, 
       edgecolor='white',bins=np.arange(0, 12, 2),cmap=cm.Greys)
ax.set_yticks(np.arange(0, 12, step=2))
plt.legend(loc = 'upper right',fontsize = 12)
ax.text(np.pi*3/4,1150,'EastGRIP Winter')
fig1.savefig(fn+'.jpg')

# reset this back to the default
plt.rc('font',size=10)
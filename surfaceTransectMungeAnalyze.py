#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:24:04 2022

read and plot the surface isotope transects as a function of time (vertically and horizontally)

then plot with the EastGRIP met data for this time period at different time scales.

@author: michaeltown
"""

# libraries
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import figureMagic as fm
from scipy.io import loadmat
import datetime as dt


# filter function
def swFilt(s):
    if s < 0:
        return np.nan
    else:
        return s

# date converting functions
def numDaysYear(y):
    y = int(y)
    return (dt.date(y,1,1)-dt.date(y-1,1,1)).days

def zeroPad(d):
    dStr = str(int(d)+1)        # add one because the datetime function goes from 1-365 for day of year
    
    if len(dStr)==1:
        return ''.join(['00',dStr])
    elif len(dStr)==2:
        return ''.join(['0',dStr])
    else:
        return dStr


# locations
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'


# read and plot the ST data, save the data in a df for later use.

years = [2016,2017,2018,2019]
cols = ["date","samplename","d18O","d18O_std","dD","dD_std","dexcess"]
df_st = pd.DataFrame(columns = cols);

i = 0

for y in years:

    fileLoc = '/home/michaeltown/work/projects/snowiso/EastGRIP'+ str(y)+'/surfaceTransect/'
    dirList = os.listdir(fileLoc)
    
    if y == 2016:
        
        df_st16 = pd.read_pickle(fileLoc+'surfaceTransectDataEastGRIP2016.pkl')
    else:
        
        dirList = [f for f in dirList if '1cm.txt' in f]
        # dirList = [f for f in dirList if '05cm' in f]
        
        df = pd.read_csv(fileLoc+dirList[0])
        df.columns = cols
        df_st = df_st.append(df)


# clean the date column and make it the df index
df_st.date = df_st.date.apply(lambda x: pd.to_datetime(x[0:10].replace('-',''),format = '%Y%m%d'))
df_st.set_index('date', inplace = True)

df_st.drop('samplename',axis = 1,inplace = True)
df_st = df_st.append(df_st16)

# plot the stake accumulation data 
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/meteo/' 
mat = loadmat(fileLoc + 'Bamboo_daily_season_HCSL.mat')

# build data frame of daily values from the mat files
col = ['stakeHeight']
df_stakes = pd.DataFrame(columns = col)

list1 = ['Bamboo_daily_2016_FracYear','Bamboo_daily_2017_FracYear','Bamboo_daily_2018_FracYear',
         'Bamboo_daily_2019_FracYear']
list2 = ['Bamboo_daily_2016_SnowHeight','Bamboo_daily_2017_SnowHeight','Bamboo_daily_2018_SnowHeight',
         'Bamboo_daily_2019_SnowHeight']
yearDict = dict(zip(list1,list2))

for k in yearDict.keys():
    
    x = dict(zip(mat[k].ravel(),mat[yearDict[k]].ravel()))
    x = pd.DataFrame.from_dict(x,orient = 'index')
    x.columns = col
    df_stakes = df_stakes.append(x)

df_stakes['year'] = np.floor(df_stakes.index)
df_stakes['dayOfYear'] = (df_stakes.index-df_stakes.year)*df_stakes.year.apply(numDaysYear)
df_stakes['dayOfYearStr'] = df_stakes.dayOfYear.apply(zeroPad)
df_stakes.index = pd.to_datetime(df_stakes.year.apply(lambda x: str(int(x)))+df_stakes.dayOfYearStr,format = '%Y%j')


for y in years:
    fig = plt.figure()
    
    plt.plot(df_stakes.index,df_stakes.stakeHeight,color = 'red',alpha = 0.5)
    plt.ylabel('snow stake height (cm)')
    plt.xlim([pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d')])
    plt.xticks(rotation = 25)
    fm.saveas(fig,figureLoc+'snowStakeAcc_daily_'+str(y))


# plot the surface transects for these time periods
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'

for y in years:
    fig = plt.figure()
    plt.plot(df_st.index[df_st.index.year == y],df_st.d18O[df_st.index.year == y],color = 'black')
    plt.xlabel('date')
    plt.ylabel('d18O')
    plt.title('surface transect for ' + str(y))
    plt.ylim([-40,-20])
    plt.xlim([pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d')])
    plt.xticks(rotation = 25)
    fm.saveas(fig,figureLoc+str(y)+'/stData_d18O_sum'+str(y))
    
for y in years:
    fig = plt.figure()
    plt.plot(df_st.index[df_st.index.year == y],df_st.dexcess[df_st.index.year == y])
    plt.xlabel('date')
    plt.ylabel('dexcess')
    plt.title('surface transect for ' + str(y))
    plt.ylim([-10,30])
    plt.xlim([pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d')])
    plt.xticks(rotation = 25)
    fm.saveas(fig,figureLoc+str(y)+'/stData_dxs_sum'+str(y))

# load the met data and plot in subplots
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
df_met = pd.read_pickle(fileLoc+'eastGRIP_PROMICEData_2016-2019.pkl')
df_met.ShortwaveRadiationDownWm2 = df_met.ShortwaveRadiationDownWm2.apply(swFilt)
df_met.ShortwaveRadiationUpWm2 = df_met.ShortwaveRadiationUpWm2.apply(swFilt)

df_met_daily = df_met.resample('D').mean()
df_met_daily['netRad'] = df_met_daily.ShortwaveRadiationDownWm2-df_met_daily.ShortwaveRadiationUpWm2+df_met_daily.LongwaveRadiationDownWm2-df_met_daily.LongwaveRadiationUpWm2


for y in years:
    fig,ax = plt.subplots(2)
    ax[1].plot(df_st.index[df_st.index.year == y],df_st.d18O[df_st.index.year == y],color = 'black')
    ax[1].set_xlabel('date')
    ax[1].set_ylabel('d18O')
    ax[1].set_ylim(-40,-20)
    ax4 = ax[1].twinx()
    p4, = ax4.plot(df_stakes.index,df_stakes.stakeHeight,color = 'Grey',alpha = 0.5)
    ax4.set_ylabel('snow acc (cm)',color = 'Grey')
    ax4.tick_params(axis='y', colors=p4.get_color(), labelsize=14)
    ax[1].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax4.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax[1].xaxis.set_tick_params(rotation=25) 
    ax4.xaxis.set_tick_params(rotation=25) 

    
    
    ax[0].plot(df_met.index,df_met.AirTemperatureC,color = 'black')
    ax[0].set_ylabel('T (oC)')
    ax[0].set_ylim(-40,10)
    ax2 = ax[0].twinx()
    p2, = ax2.plot(df_met.index,df_met.WindSpeedms,color = 'Grey',alpha = 0.5)
    ax2.set_ylabel('wind speed (m/s)',color = 'Grey')
    ax3 = ax[0].twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    p3, = ax3.plot(df_met_daily.index,df_met_daily.netRad,color = 'red',alpha = 0.5)
    ax3.set_ylabel('net rad (W/m2)',color = 'red')
    ax[0].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax2.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax3.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=14)
    ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    ax[0].set_xticklabels('')
    ax2.set_xticklabels('')
    ax3.set_xticklabels('')
    ax3.set_ylim(-40,40)
    fm.saveas(fig,figureLoc+str(y)+'/stData_d18Omet_sum'+str(y))

    
for y in years:
    fig,ax = plt.subplots(2)
    ax[1].plot(df_st.index[df_st.index.year == y],df_st.dexcess[df_st.index.year == y])
    ax[1].set_xlabel('date')
    ax[1].set_ylabel('dxs')
    ax[1].set_ylim(-10,30)
    ax[1].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax4 = ax[1].twinx()
    p4, = ax4.plot(df_stakes.index,df_stakes.stakeHeight,color = 'Grey',alpha = 0.5)
    ax[1].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax4.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax[1].xaxis.set_tick_params(rotation=25) 
    ax4.xaxis.set_tick_params(rotation=25) 
#    ax[1].set_xticklabels(ax[1].get_xticks(), rotation =25)

    #ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    
    ax[0].plot(df_met.index,df_met.AirTemperatureC,color = 'black')
    ax[0].set_ylabel('T (oC)')
    ax[0].set_ylim(-40,10)
    ax2 = ax[0].twinx()
    p2, = ax2.plot(df_met.index,df_met.WindSpeedms,color = 'Grey',alpha = 0.5)
    ax2.set_ylabel('wind speed (m/s)',color = 'Grey')
    ax3 = ax[0].twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    p3, = ax3.plot(df_met_daily.index,df_met_daily.netRad,color = 'red',alpha = 0.5)
    ax3.set_ylabel('net rad (W/m2)',color = 'red')
    ax[0].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax2.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax3.set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
    ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=14)
    ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    ax[0].set_xticklabels('')
    ax2.set_xticklabels('')
    ax3.set_xticklabels('')
    ax3.set_ylim(-40,40)
    fm.saveas(fig,figureLoc+str(y)+'/stData_dxsMet_sum'+str(y))

# save the ST data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'

df_st.to_pickle(fileLoc+'surfaceTransectDataEastGRIP2016-2019.pkl')



# load 0.5 cm, 1 cm, and, 2 cm data from 2018 to see the impact on dxs and d18O by the depth
# of the measurement

years = [2018,2019]
fileStrs = ['05cm.txt','1cm.txt','2cm.txt']
cols = ["date","samplename","d18O","d18O_std","dD","dD_std","dexcess","type"]
df_stDepth = pd.DataFrame(columns = cols)
alphas = np.arange(1,0.1,-1/(len(fileStrs)+1))

for y in years:
    fileLoc = '/home/michaeltown/work/projects/snowiso/EastGRIP'+ str(y)+'/surfaceTransect/'
    dirList = os.listdir(fileLoc)
    
    for f in fileStrs:
        
        fileName = [fi for fi in dirList if f in fi]
        
        df = pd.read_csv(fileLoc+fileName[0])
        df['type'] = f
        df.columns = cols
        df.columns = cols
        df_stDepth = df_stDepth.append(df)
    
    
df_stDepth.date = df_stDepth.date.apply(lambda x: pd.to_datetime(x[0:10].replace('-',''),format = '%Y%m%d'))
df_stDepth.set_index('date', inplace = True)

    
    # plot the surface transect data as a function of time for 2018 using the depth as a sorting value


for y in years:    
    fig,ax = plt.subplots(2)
    # clean the date column and make it the df index
    
    i = 0
    for f in fileStrs:
        
        ax[0].plot(df_stDepth[(df_stDepth.index.year == y)&(df_stDepth.type == f)].index,df_stDepth[(df_stDepth.index.year == y)&(df_stDepth.type == f)].d18O,
                   color = 'black',alpha = alphas[i],label = f)
        ax[0].set_ylabel('d18O')
        ax[0].set_ylim(-40,-20)
        ax[0].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
        ax[0].set_xticklabels('')
        ax[0].legend()
    
        ax[1].plot(df_stDepth[(df_stDepth.index.year == y)&(df_stDepth.type == f)].index,df_stDepth[(df_stDepth.index.year == y)&(df_stDepth.type == f)].dexcess,
                   color = 'blue',alpha = alphas[i],label = f)
        ax[1].set_xlabel('date')
        ax[1].set_ylabel('dxs')
        ax[1].set_ylim(-10,30)
        ax[1].set_xlim(pd.to_datetime(str(y)+'0501',format = '%Y%m%d'),pd.to_datetime(str(y)+'0831',format = '%Y%m%d'))
        plt.xlabel('date')
        plt.xticks(rotation = 25)
        ax[1].legend()
    
        i += 1
    
    fm.saveas(fig, figureLoc+str(y)+'/stData_dxsMet_sum'+str(y))

## stats check
print('05cm.txt d18O mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '05cm.txt'].d18O),2)) + '+/-' + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '05cm.txt'].d18O),2)))
print('1cm.txt d18O mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '1cm.txt'].d18O),2)) + '+/-' + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '1cm.txt'].d18O),2)))
print('2cm.txt d18O mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '2cm.txt'].d18O),2)) + '+/-'  + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '2cm.txt'].d18O),2)))

print('05cm.txt dxs mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '05cm.txt'].dexcess),2)) + '+/-'  + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '05cm.txt'].dexcess),2)))
print('1cm.txt dxs mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '1cm.txt'].dexcess),2)) + '+/-'  + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '1cm.txt'].dexcess),2)))
print('2cm.txt dxs mean, std = ' + str(np.round(np.mean(df_stDepth[df_stDepth.type == '2cm.txt'].dexcess),2)) + '+/-'  + 
      str(np.round(np.std(df_stDepth[df_stDepth.type == '2cm.txt'].dexcess),2)))

print('2016: ')
print(np.mean(df_st16))
print('2017: ')
print(np.mean(df_st[df_st.index.year == 2017]))
print('2018: ')
print(np.mean(df_st[df_st.index.year == 2018]))
print('2019: ')
print(np.mean(df_st[df_st.index.year == 2019]))


print('2016: ')
print(np.mean(df_st16))
for y in years:
    print(str(y) + ': ')
    beg = dt.date(y,6,10)
    end = dt.date(y,8,5)
    print(np.mean(df_st[(df_st.index.date > beg)&(df_st.index.date < end)]))

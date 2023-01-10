#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:17:48 2022

version three of the post-dep modification paper figures

@author: michaeltown
"""


# libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable
import figureMagic as fm
import EastGRIPprocs as eg
import calendar
import datetime as datetime
from datetime import timedelta
from scipy.io import loadmat
import datetime as dt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def countFunc(n):
    
    return len(n)-np.count_nonzero(np.isnan(n))

# change fonts 
mpl.rcParams['font.family'] ='Helvetica'

#symbols
d18Osym = '$\delta^{18}$O ($^{o}/_{oo}$)'
dDsym = '$\delta$D ($^{o}/_{oo}$)'
dxssym = 'd-excess  ($^{o}/_{oo}$)'
d18Osym2 = '$\delta^{18}$O'


# basic inputs
numCores2019 = 25
numCores2018 = 35
numCores2017 = 32           # dropped position number 4 from all stats here
sl = 2                      # sigma level

coreNumDict = dict(zip([2017,2018,2019],[numCores2017,numCores2018,numCores2019]))

# file locations
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/postDepDataPub2023/'

vers = input('Which version of the data to plot (m = mean acc model, c = explicit acc model): ')

if vers.lower() == 'm':
    fileNameDepth = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t2.pkl'
    fileNameTime = 'eastGRIP_SCisoData_2017-2019_tindex_accExt_t2.pkl'
    figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/postDepDataPub2023_m/'
    dxsMeanTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_mean_accExt_t2.pkl'
    dxsStdTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_std_accExt_t2.pkl'
    dxsMinTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_min_accExt_t2.pkl'
    dxsMaxTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_max_accExt_t2.pkl'
    d18OMeanTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_mean_accExt_t2.pkl'
    d18OStdTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_std_accExt_t2.pkl'
    d18OMinTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_min_accExt_t2.pkl'
    d18OMaxTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_max_accExt_t2.pkl'

elif vers.lower() == 'c':
    fileNameDepth = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t3.pkl'
    fileNameTime = 'eastGRIP_SCisoData_2017-2019_tindex_accExt_t3.pkl'
    figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/postDepDataPub2023_c/'
    dxsMeanTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_mean_accExt_t3.pkl'
    dxsStdTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_std_accExt_t3.pkl'
    dxsMinTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_min_accExt_t3.pkl'
    dxsMaxTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_dxs_tindex_max_accExt_t3.pkl'
    d18OMeanTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_mean_accExt_t3.pkl'
    d18OStdTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_std_accExt_t3.pkl'
    d18OMinTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_min_accExt_t3.pkl'
    d18OMaxTimeProfileFileName = 'eastGRIP_SCisoData_2017-2019_d18O_tindex_max_accExt_t3.pkl'   

# accumulation data
fileLocMeteo = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
dataFilenameAcc = 'Sonic_mean2016-2019_HCSL.txt'

# surface transect data 


# snow stake data

# plot the stake accumulation data 
figureLocMeteo = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/meteo/' 
mat = loadmat(fileLocMeteo + 'Bamboo_daily_season_HCSL.mat')

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
df_stakes.to_pickle(fileLoc+'EastGRIPsnowStakes2016_2019.pkl')

# figure limits
d18Omin = -50
d18Omax = -20
dxsmin = -10
dxsmax = 20

######
# load data
######
# load isotope data
df_isod = pd.read_pickle(fileLoc+fileNameDepth)
df_isot = pd.read_pickle(fileLoc+fileNameTime)


# load the PROMICE data
promiceLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
promiceDataFileName= 'eastGRIP_PROMICEData_2016-2019.pkl';
df_promice = pd.read_pickle(promiceLoc + promiceDataFileName)

df_acc = pd.read_csv(fileLocMeteo+dataFilenameAcc,sep='\t')

# load the surface trasect data here


#####
# plot figures
#####

# plot figures by year
dvals = df_isod.index.get_level_values(0).unique()
coreIDs = df_isod.index.get_level_values(1).unique()
dates = df_isod.index.get_level_values(2).unique()

years = dates.year.unique()

coreIDsAll = df_isod.index.get_level_values(1)
yearsAll = df_isod.index.get_level_values(2).year

############
# mean profiles by depth, d18O, dxs and stacked vertically
############

df_acc['year'] = np.floor(df_acc.Julian_Sonic_mean)
df_acc['julDay'],df_acc['decDay'] = eg.decimalToDaysInYear(df_acc.year.astype(int).apply(calendar.isleap), df_acc.Julian_Sonic_mean-df_acc.year)

df_acc['date'] = df_acc.year.astype(int).astype(str)+'-'+df_acc.julDay.astype(int).astype(str)
df_acc['datetime'] =pd.to_datetime(df_acc.date,format = '%Y-%j')

jul2019 = datetime.datetime.strptime('20190716', '%Y%m%d')
jul2018 = datetime.datetime.strptime('20180716', '%Y%m%d')
jul2017 = datetime.datetime.strptime('20170716', '%Y%m%d')
acc2017 = df_acc[df_acc.datetime == jul2018].Sonic_cumsum_snow_height.values-df_acc[df_acc.datetime == jul2017].Sonic_cumsum_snow_height.values
acc2018 = df_acc[df_acc.datetime == jul2019].Sonic_cumsum_snow_height.values-df_acc[df_acc.datetime == jul2018].Sonic_cumsum_snow_height.values

acc2017 = int(np.round(acc2017,0))
acc2018 = int(np.round(acc2018,0))

lnstyDict = dict(zip([2017,2018,2019],['-','-','-']))
hatchDict = dict(zip([2017,2018,2019],['///','\\/','']))
lineclr_d18O = dict(zip([2017,2018,2019],['SlateBlue','Blue','DarkBlue']))
lineclr_dxs = dict(zip([2017,2018,2019],['grey','dimgrey','Black']))
depthVals = df_isod.index.get_level_values(0).unique()
depthValsAll = df_isod.index.get_level_values(0)
coreIDsAlld = df_isod.index.get_level_values(1)
yearsAlld = df_isod.index.get_level_values(2)

df_isod_mean_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isod_std_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isod_num_d18O = pd.DataFrame(columns = [2017,2018,2019])

# d18O averaged profile
fig = plt.subplots()
plt.subplot(1,7,(1,2))
for y in years:
    if y == 2017:
        depthValsP = depthVals + acc2017+ acc2018
        coreIDsAlld = [1,2,3,5]
    elif y == 2018:
        depthValsP = depthVals + acc2018
        coreIDsAlld = df_isod.index.get_level_values(1)
    else:
        depthValsP = depthVals
        coreIDsAlld = df_isod.index.get_level_values(1)


    df_isod_mean_d18O[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.mean)
    df_isod_std_d18O[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.std)
    df_isod_num_d18O[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(countFunc)
    

    # plot the profile data
    se = eg.seComp(sl,df_isod_std_d18O[y],np.sqrt(df_isod_num_d18O[y]))

    plt.plot(df_isod_mean_d18O[y],-depthValsP,color = lineclr_d18O[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(-depthValsP,df_isod_mean_d18O[y]-se,df_isod_mean_d18O[y]+se,color = 'Blue',
                      alpha = 0.2,hatch = hatchDict[y])

plt.xlabel(d18Osym)
plt.ylabel('Depth (cm)')
plt.ylim([-200,20])
for y in np.arange(-160,20,40):
    plt.plot([-50,-20],[y,y],':',color = 'Grey')
    plt.xticks([-5,0,5])
plt.xticks([-50,-40,-30])
plt.xlim([d18Omin,d18Omax])
fm.cleanLegend('lower right')
plt.yticks(np.arange(-200,40,20))

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,7,3)
prof2017to2019 = pd.DataFrame(df_isod_mean_d18O[2017].copy())
prof2018to2019 = pd.DataFrame(df_isod_mean_d18O[2018].copy())
prof2019 = pd.DataFrame(df_isod_mean_d18O[2019].copy())

prof2017to2019['newDepth'] = depthVals+acc2017+acc2018
prof2018to2019['newDepth'] = depthVals+acc2018

prof2017to2019 = prof2017to2019.set_index('newDepth')
prof2018to2019 = prof2018to2019.set_index('newDepth')
nDepthVals2018 = prof2018to2019.index.values
nDepthVals2017 = prof2017to2019.index.values

dfMeanDepth_d18O = pd.DataFrame(index = np.arange(-20,191,1),columns = [2017,2018,2019])

for i in prof2017to2019.index:
    dfMeanDepth_d18O.loc[i,2017] = prof2017to2019.loc[i,2017]

for i in prof2018to2019.index:
    dfMeanDepth_d18O.loc[i,2018] = prof2018to2019.loc[i,2018]

for i in prof2019.index:
    dfMeanDepth_d18O.loc[i,2019] = prof2019.loc[i,2019] 

plt.plot(dfMeanDepth_d18O[2018]-dfMeanDepth_d18O[2017],-dfMeanDepth_d18O.index, label = '2018-2017')
plt.plot(dfMeanDepth_d18O[2019]-dfMeanDepth_d18O[2017],-dfMeanDepth_d18O.index, label = '2019-2017')
plt.plot(dfMeanDepth_d18O[2019]-dfMeanDepth_d18O[2018],-dfMeanDepth_d18O.index, label = '2019-2018')
plt.xlabel('$\Delta$'+d18Osym)
plt.xlim([-12,12])
plt.xticks([-10,0,10])
plt.ylim([-200,20])
plt.yticks(np.arange(-200,40,20),alpha = 0)
for y in np.arange(-160,20,40):
    plt.plot([-12,12],[y,y],':',color = 'Grey')
plt.plot([0,0],[-200,20],':',color = 'black',linewidth = 0.25)


#####
# d18O Age depth profile
#####

lnstyDict = dict(zip([2017,2018,2019],['-','-','-']))
hatchDict = dict(zip([2017,2018,2019],['///','.','']))
lineclr = dict(zip([2017,2018,2019],['SlateBlue','Blue','DarkBlue']))


timeValsAll = df_isot.index.get_level_values(0)
timeVals = df_isot.index.get_level_values(0).unique()
yearsAllt = df_isot.index.get_level_values(2)
df_isot_mean_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isot_std_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isot_num_d18O = pd.DataFrame(columns = [2017,2018,2019])

plt.subplot(1,7,(5,6))
for y in years:
    if y == 2017:
        coreIDsAllt = [1,2,3,5]
    else:
        coreIDsAllt = df_isot.index.get_level_values(1)


    df_isot_mean_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.mean)
    df_isot_std_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.std)
    df_isot_num_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(countFunc)

    # plot the profile data
    se = eg.seComp(sl,df_isot_std_d18O[y],np.sqrt(df_isot_num_d18O[y]))

    plt.plot(df_isot_mean_d18O[y],timeVals,color = lineclr[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(timeVals,df_isot_mean_d18O[y]-se,df_isot_mean_d18O[y]+se,color = 'Blue',alpha = 0.2,
                      hatch = hatchDict[y])

plt.xlabel(d18Osym)
plt.xlim([d18Omin,d18Omax])
plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),np.max(timeVals)])

plt.xticks([-50,-40,-30])

# set horizontal lines at summer peaks
yline = np.asarray([])
for i in np.arange(2015,2021,1):
    yline = np.append(yline,pd.to_datetime(str(i)+'0731',format = '%Y%m%d'))

for y in yline:
    plt.plot([d18Omin,d18Omax],[y,y],':',color = 'Grey')
plt.ylabel('Age depth (years)')

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,7,7)
plt.plot(df_isot_mean_d18O[2018]-df_isot_mean_d18O[2017],timeVals, label = '2018-2017')
plt.plot(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2017],timeVals, label = '2019-2017')
plt.plot(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2018],timeVals, label = '2019-2018')
plt.xlabel('$\Delta'+d18Osym)
plt.xlim([-12,12])
plt.xticks([-10,0,10])
plt.yticks(color = 'White',alpha = 0)
plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),np.max(timeVals)])
fm.cleanLegend('lower right')
plt.plot([0,0],[pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)],'--',linewidth = 0.25,color = 'Black')
for y in yline:
    plt.plot([-12,12],[y,y],':',color = 'Grey')
fm.saveas(fig[0], figureLoc+'d18O_depthProfileStacked_td_2017_2019_'+vers)


# time-weighted residual values from the mean profiles
print('time-weighted change of d18O from 2018-2017: ' + 
      str(np.round(np.sum(df_isot_mean_d18O[2018]-df_isot_mean_d18O[2017]),2)))
print('time-weighted change of d18O from 2019-2017: ' + 
      str(np.round(np.sum(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2017]),2)))
print('time-weighted change of d18O from 2019-2018: ' + 
      str(np.round(np.sum(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2018]),2)))

# depth-weighted residual values from the mean profiles
print('depth-weighted change of d18O from 2018-2017: ' + 
      str(np.round(np.sum(df_isod_mean_d18O[2018]-df_isod_mean_d18O[2017]),2)))
print('depth-weighted change of d18O from 2019-2017: ' + 
      str(np.round(np.sum(df_isod_mean_d18O[2019]-df_isod_mean_d18O[2017]),2)))
print('depth-weighted change of d18O from 2019-2018: ' +
      str(np.round(np.sum(df_isod_mean_d18O[2019]-df_isod_mean_d18O[2018]),2)))

############
# dxs age-depth profiles
############


df_isod_mean_dxs = pd.DataFrame()
df_isod_std_dxs = pd.DataFrame()
df_isod_num_dxs = pd.DataFrame(columns = [2017,2018,2019])

fig = plt.subplots()
plt.subplot(1,7,(1,2))
for y in years:
    if y == 2017:
        depthValsP = depthVals + acc2017+ acc2018 
        coreIDsAlld = [1,2,3,5]
    elif y == 2018:
        depthValsP = depthVals + acc2018
        coreIDsAlld = df_isod.index.get_level_values(1)
    else:
        depthValsP = depthVals
        coreIDsAlld = df_isod.index.get_level_values(1)


    df_isod_mean_dxs[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.mean)
    df_isod_std_dxs[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.std)
    df_isod_num_dxs[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(countFunc)

    # plot the profile data
    se = eg.seComp(sl,df_isod_std_dxs[y],np.sqrt(df_isod_num_dxs[y]))

    plt.plot(df_isod_mean_dxs[y],-depthValsP,color = lineclr_dxs[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(-depthValsP,df_isod_mean_dxs[y]-se,df_isod_mean_dxs[y]+se,color = 'Grey',alpha = 0.2,
                      hatch = hatchDict[y])

plt.xlabel(dxssym)
plt.xlim([-10,20])
plt.xticks([-10,0,10])
plt.plot([0,0],[-200,20],':',color = 'Black',linewidth = 0.25)
plt.ylim([-200,20])
plt.yticks(np.arange(-200,40,20))
plt.ylabel('Depth (cm)')
for y in np.arange(-160,20,40):
    plt.plot([-10,30],[y,y],':',color = 'Grey')
fm.cleanLegend('lower right')

# plot the residual profile data in an adjacent subplot 
prof2017to2019 = pd.DataFrame(df_isod_mean_dxs[2017].copy())
prof2018to2019 = pd.DataFrame(df_isod_mean_dxs[2018].copy())
prof2019 = pd.DataFrame(df_isod_mean_dxs[2019].copy())

prof2017to2019['newDepth'] = depthVals+acc2017+acc2018
prof2018to2019['newDepth'] = depthVals+acc2018

prof2017to2019 = prof2017to2019.set_index('newDepth')
prof2018to2019 = prof2018to2019.set_index('newDepth')
nDepthVals2018 = prof2018to2019.index.values
nDepthVals2017 = prof2017to2019.index.values

dfMeanDepth_dxs = pd.DataFrame(index = np.arange(-20,191,1),columns = [2017,2018,2019])

for i in prof2017to2019.index:
    dfMeanDepth_dxs.loc[i,2017] = prof2017to2019.loc[i,2017]

for i in prof2018to2019.index:
    dfMeanDepth_dxs.loc[i,2018] = prof2018to2019.loc[i,2018]

for i in prof2019.index:
    dfMeanDepth_dxs.loc[i,2019] = prof2019.loc[i,2019] 

plt.subplot(1,7,3)
plt.plot(dfMeanDepth_dxs[2018]-dfMeanDepth_dxs[2017],-dfMeanDepth_dxs.index, label = "'18-'17")
plt.plot(dfMeanDepth_dxs[2019]-dfMeanDepth_dxs[2017],-dfMeanDepth_dxs.index, label = "'19-'17")
plt.plot(dfMeanDepth_dxs[2019]-dfMeanDepth_dxs[2018],-dfMeanDepth_dxs.index, label = "'19-'18")
plt.xlabel('$\Delta$'+dxssym)
plt.xlim([-12,12])
plt.xticks([-10,0,10])
plt.ylim([-200,20])
plt.yticks(np.arange(-200,40,20),alpha = 0)
plt.plot([0,0],[-200,20],':',color = 'Black',linewidth = 0.25)
for y in np.arange(-160,20,40):
    plt.plot([-12,12],[y,y],':',color = 'Grey')

timeValsAll = df_isot.index.get_level_values(0)
timeVals = df_isot.index.get_level_values(0).unique()
coreIDsAllt = df_isot.index.get_level_values(1)
yearsAllt = df_isot.index.get_level_values(2)
df_isot_mean_dxs = pd.DataFrame(columns = [2017,2018,2019])
df_isot_std_dxs = pd.DataFrame(columns = [2017,2018,2019])
df_isot_num_dxs = pd.DataFrame(columns = [2017,2018,2019])

plt.subplot(1,7,(5,6))
for y in years:
    if y == 2017:
        coreIDsAllt = [1,2,3,5]
    else:
        coreIDsAllt = df_isot.index.get_level_values(1)

    df_isot_mean_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.mean)
    df_isot_std_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.std)
    df_isot_num_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(countFunc)

    # plot the profile data
    se = eg.seComp(sl,df_isot_std_dxs[y],np.sqrt(df_isot_num_dxs[y]))

    plt.plot(df_isot_mean_dxs[y],timeVals,color = lineclr_dxs[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(timeVals,df_isot_mean_dxs[y]-se,df_isot_mean_dxs[y]+se,color = 'Grey',alpha = 0.2,
                      hatch = hatchDict[y])
plt.xlabel(dxssym)
plt.xlim([dxsmin,dxsmax])
plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),np.max(timeVals)])
plt.ylabel('Age depth (years)')
plt.plot([0,0],[-200,20],':',color = 'Black',linewidth = 0.25)
plt.xticks([-10,0,10])

# set horizontal lines at summer peaks
yline = np.asarray([])
for i in np.arange(2015,2021,1):
    yline = np.append(yline,pd.to_datetime(str(i)+'0731',format = '%Y%m%d'))

for y in yline:
    plt.plot([dxsmin,dxsmax],[y,y],':',color = 'Grey')


for s in se: 
    if(~np.isnan(s)): 
        print(s)
# plot the residual profile data in an adjacent subplot 
plt.subplot(1,7,7)
plt.plot(df_isot_mean_dxs[2018]-df_isot_mean_dxs[2017],timeVals, label = "'18-'17")
plt.plot(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2017],timeVals, label = "'19-'17")
plt.plot(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2018],timeVals, label = "'19-'18")
plt.xlabel('$\Delta$' + dxssym)
plt.xlim([-12,12])
plt.xticks([-10,0,10])
plt.yticks(color = 'White',alpha = 0)
plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),np.max(timeVals)])
fm.cleanLegend('lower right')
plt.plot([0,0],[pd.to_datetime('20141101',format = '%Y%m%d'),np.max(timeVals)],'--',linewidth = 0.25,color = 'Black')
for y in yline:
    plt.plot([-12,12],[y,y],':',color = 'Grey')

fm.saveas(fig[0], figureLoc+'dxs_depthProfileStacked_td_2017_2019_'+vers)


# time-weighted residual values from the mean profiles
print('time-weighted change of d18O from 2018-2017: ' + 
      str(np.round(np.sum(df_isot_mean_dxs[2018]-df_isot_mean_dxs[2017]),2)))
print('time-weighted change of d18O from 2019-2017: ' + 
      str(np.round(np.sum(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2017]),2)))
print('time-weighted change of d18O from 2019-2018: ' + 
      str(np.round(np.sum(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2018]),2)))

# depth-weighted residual values from the mean profiles
print('depth-weighted change of d18O from 2018-2017: ' + 
      str(np.round(np.sum(df_isod_mean_dxs[2018]-df_isod_mean_dxs[2017]),2)))
print('depth-weighted change of d18O from 2019-2017: ' + 
      str(np.round(np.sum(df_isod_mean_dxs[2019]-df_isod_mean_dxs[2017]),2)))
print('depth-weighted change of d18O from 2019-2018: ' +
      str(np.round(np.sum(df_isod_mean_dxs[2019]-df_isod_mean_dxs[2018]),2)))

############
# mean contour time/depth series
############
#d18O

df_p_ave_d18O = {}
df_p_std_d18O = {}


for y in years:

    if y == 2017:           # drops position 4 which had some issues with accumulation recording
        coreIDsAll = [1,2,3,5]
        
    df_p_ave = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].groupby(['depth','date']).d18O.apply(np.mean)
    df_p_ave = df_p_ave.reset_index(); 
    df_p_ave = df_p_ave.pivot(*df_p_ave.columns)                  # what does this * do here? something magical, ok pass a number of things to pivot

    df_p_std = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].groupby(['depth','date']).d18O.apply(np.std)
    df_p_std = df_p_std.reset_index(); 
    df_p_std = df_p_std.pivot(*df_p_std.columns)

    # drop columns with all nan values
    df_p_ave = df_p_ave.dropna(axis = 1, how = 'all')
    df_p_std = df_p_std.dropna(axis = 1, how = 'all')

    # which dates are remaining?
    datesLeft = df_p_ave.columns

    if y == 2017:           # drops the paired dates in 2017 where samples were not all taken on the same day
        df_p_ave.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        df_p_ave.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
        df_p_std.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        df_p_std.drop(axis = 1, columns = datesLeft[8:10], inplace = True)

        datesLeft = df_p_ave.columns

    df_p_ave_d18O[y] = df_p_ave;
    df_p_std_d18O[y] = df_p_std;




    # set date range for the plot
    d1 = pd.to_datetime(str(y)+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(y)+'0815',format = '%Y%m%d')
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'Blues',vmin = d18Omin, vmax = -20)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')    
    ax2[1].set(xlim = [d1,d2])
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(d18Omin, d18Omax+5, 5))
    cbar.set_ticks(np.arange(d18Omin,d18Omax,5))
    plt.ylim(d18Omin,10)
    plt.ylabel('Depth (cm)')
    plt.xticks(rotation = 25)
    for date in datesLeft:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(datesLeft[1],-40,'mean d18O')
    fm.saveas(fig1, figureLoc+'d18O_depthTimeContour_mean_'+str(y))

    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_p_std.columns, -df_p_std.index, df_p_std.values, cmap = 'Greys',vmin = 0, vmax = 10)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')    
    ax2[1].set(xlim = [d1,d2])
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(0, 11, 2))
    cbar.set_ticks(np.arange(0,10,2))
    plt.ylim(d18Omin,10)
    plt.ylabel('Depth (cm)')
    plt.xticks(rotation = 25)
    for date in datesLeft:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(datesLeft[1],-40,'std d18O')
    fm.saveas(fig1, figureLoc+'d18O_depthTimeContour_std_'+str(y))

#dxs
df_p_ave_dxs = {}
df_p_std_dxs = {}

for y in years:

    if y == 2017:
        coreIDsAll = [1,2,3,5]
    

    df_p_ave = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].groupby(['depth','date']).dexcess.apply(np.mean)
    df_p_ave = df_p_ave.reset_index(); 
    df_p_ave = df_p_ave.pivot(*df_p_ave.columns)                  # what does this * do here? something magical, ok pass a number of things to pivot

    df_p_std = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].groupby(['depth','date']).dexcess.apply(np.std)
    df_p_std = df_p_std.reset_index(); 
    df_p_std = df_p_std.pivot(*df_p_std.columns)


    # drop columns with all nan values
    df_p_ave = df_p_ave.dropna(axis = 1, how = 'all')
    df_p_std = df_p_std.dropna(axis = 1, how = 'all')

    # which dates are remaining?
    datesLeft = df_p_ave.columns
    if y == 2017:           # drops the paired dates in 2017 where samples were not all taken on the same day
        df_p_ave.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        df_p_ave.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
        df_p_std.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        df_p_std.drop(axis = 1, columns = datesLeft[8:10], inplace = True)

        datesLeft = df_p_ave.columns

    df_p_std_dxs[y] = df_p_std;
    df_p_ave_dxs[y] = df_p_ave;


    d1 = pd.to_datetime(str(y)+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(y)+'0815',format = '%Y%m%d')
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'bwr',vmin = -10, vmax = 25)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')
    ax2[1].set(xlim = [d1,d2])
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-10, 25+5, 5))
    cbar.set_ticks(np.arange(-10,25,5))
    plt.ylim(d18Omin,5)
    plt.xlabel('date')
    plt.ylabel('Depth (cm)')
    for date in datesLeft:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.xticks(rotation = 25)
    plt.text(datesLeft[1],-40,'mean dxs')
    fm.saveas(fig1, figureLoc+'dxs_depthTimeContour_mean_'+str(y))


    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_p_std.columns, -df_p_std.index, df_p_std.values, cmap = 'Greys',vmin = 0, vmax = 10)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')    
    ax2[1].set(xlim = [d1,d2])
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(0, 11, 2))
    cbar.set_ticks(np.arange(0,10,2))
    plt.ylim(d18Omin,10)
    plt.ylabel('Depth (cm)')
    plt.xticks(rotation = 25)
    for date in datesLeft:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(datesLeft[1],-40,'std dxs')
    fm.saveas(fig1, figureLoc+'dxs_depthTimeContour_std_'+str(y))


############
# contour time/depth series by coreID
############
#d18O
for y in years:
    
    if (y == 2018):
        coreIDs = np.arange(1,7,1)
    else:
        coreIDs = np.arange(1,6,1)

    for c in coreIDs:
        df_p_ave = df_isod.loc[(dvals,c,yearsAll == y)].groupby(['depth','date']).d18O.apply(np.mean)
        df_p_ave = df_p_ave.reset_index(); 
        df_p_ave = df_p_ave.pivot(*df_p_ave.columns)                  # what does this * do here? something magical, ok pass a number of things to pivot
    
        # drop columns with all nan values
        df_p_ave = df_p_ave.dropna(axis = 1, how = 'all')
    
        # which dates are remaining?
        datesLeft = df_p_ave.columns
        # if y == 2017:           # drops the paired dates in 2017 where samples were not all taken on the same day
        #     df_p_ave.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        #     df_p_ave.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
        #     df_p_std.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        #     df_p_std.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
    
        #     datesLeft = df_p_ave.columns
    
        # set date range for the plot
        d1 = pd.to_datetime(str(y)+'0501',format = '%Y%m%d')
        d2 = pd.to_datetime(str(y)+'0815',format = '%Y%m%d')
        fig1, ax2  = plt.subplots(2,1);
        cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'Blues',vmin = d18Omin, vmax = -20)
        ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
        ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
        ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
        ax2[0].set_ylabel('T (oC)')
        ax2[0].set(xlim = [d1,d2])
        ax2[0].set_xticklabels('')
        ax2[1].set(xlim = [d1,d2])
        cbar = fig1.colorbar(
            ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
            ticks=range(d18Omin, -20+5, 5))
        cbar.set_ticks(np.arange(d18Omin,-20,5))
        plt.ylim(d18Omin,10)
        plt.ylabel('Depth (cm)')
        plt.xticks(rotation = 25)
        for date in datesLeft:
            plt.text(date,-20,' |')
            plt.text(date,-18,'^')
        plt.text(datesLeft[1],-40,'d18O for p ' + str(c))
        fm.saveas(fig1, figureLoc+'d18O_depthTimeContour_'+str(y)+'_p_'+ str(c))

#dxs
for y in years:
    if (y == 2018):
        coreIDs = np.arange(1,7,1)
    else:
        coreIDs = np.arange(1,6,1)

    for c in coreIDs:
        df_p_ave = df_isod.loc[(dvals,c,yearsAll == y)].groupby(['depth','date']).dexcess.apply(np.mean)
        df_p_ave = df_p_ave.reset_index(); 
        df_p_ave = df_p_ave.pivot(*df_p_ave.columns)                  # what does this * do here? something magical, ok pass a number of things to pivot
    
        # drop columns with all nan values
        df_p_ave = df_p_ave.dropna(axis = 1, how = 'all')
    
        # which dates are remaining?
        datesLeft = df_p_ave.columns
        # if y == 2017:           # drops the paired dates in 2017 where samples were not all taken on the same day
        #     df_p_ave.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        #     df_p_ave.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
        #     df_p_std.drop(axis = 1, columns = datesLeft[5:7], inplace = True)
        #     df_p_std.drop(axis = 1, columns = datesLeft[8:10], inplace = True)
    
        #     datesLeft = df_p_ave.columns    

        # set date range for the plot
        d1 = pd.to_datetime(str(y)+'0501',format = '%Y%m%d')
        d2 = pd.to_datetime(str(y)+'0815',format = '%Y%m%d')

        # which dates are remaining?
        datesLeft = df_p_ave.columns
        fig1, ax2  = plt.subplots(2,1);
        cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'bwr',vmin = -10, vmax = 25)
        ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
        ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
        ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
        ax2[0].set_ylabel('T (oC)')
        ax2[0].set(xlim = [d1,d2])
        ax2[0].set_xticklabels('')
        ax2[1].set(xlim = [d1,d2])
        cbar = fig1.colorbar(
            ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
            ticks=range(dxsmin, dxsmax+5, 5))
        cbar.set_ticks(np.arange(-10,25,5))
        plt.ylim(-50,10)
        plt.ylabel('Depth (cm)')
        plt.xticks(rotation = 25)
        for date in datesLeft:
            plt.text(date,-20,' |')
            plt.text(date,-18,'^')
        plt.text(datesLeft[1],-40,'dxs for p ' + str(c))
        fm.saveas(fig1, figureLoc+'dxs_depthTimeContour_'+str(y)+'_p_'+ str(c))


############
# contour megaplot
############

dictXfntClr = dict(zip(years,xFntClr))
dictFntSz = dict(zip(years,fntSize))

# creating grid for subplots
fig = plt.figure()
 
# temp/accumulation plots
locations = [0,2,4]
years = [2017,2018,2019]
locDict = dict(zip(locations,years))

for k in locDict:
    axta = plt.subplot2grid(shape=(5, 6), loc=(0, k), colspan=2)
    d1 = pd.to_datetime(str(locDict[k])+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(locDict[k])+'0815',format = '%Y%m%d')



    axta.plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC,
           color = 'Red', alpha = 0.5, linewidth = 1)
    axtat = axta.twinx()
    axtat.plot(df_stakes[(df_stakes.index>d1)&(df_stakes.index<d2)].index,df_stakes[(df_stakes.index>d1)&(df_stakes.index<d2)].stakeHeight,
            color = 'Black', alpha = 0.5, linewidth = 2)
    if locDict[k] == 2017:
        axta.set_ylabel('T$_{2m}$ $^{o}$C',color = 'red')
        axtat.set_yticklabels([])
        axta.tick_params(axis = 'y',colors = 'Red')


    if locDict[k] == 2018:
        axta.set_yticklabels([])
        axtat.set_yticklabels([])
        
    if locDict[k] == 2019:
        axtat.set_ylabel('Acc (cm)',color = 'Grey')
        axta.set_yticklabels([])
        axta.tick_params(axis = 'y',colors = 'Grey')

    
    axta.set_xticklabels([])
    axta.set_yticks([-40,-20,0])
    axta.set_ylim([-50,5])
    axtat.set_ylim([-5,22])
    axtat.set_yticks([0,10,20])



    axta.set_xlim([d1,d2])
    axtat.set_xlim([d1,d2])
    xtcks = axta.get_xticks()
    axta.set_xticks(xtcks[0::2])


# d18O contour plots

for k in locDict:
    axd18O = plt.subplot2grid(shape=(5, 6), loc=(1, k), colspan=2, rowspan = 2)
    d1 = pd.to_datetime(str(locDict[k])+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(locDict[k])+'0815',format = '%Y%m%d')
    d22 = pd.to_datetime(str(locDict[k])+'0725',format = '%Y%m%d')
    datesLeft = df_p_ave_d18O[locDict[k]].columns

    cntr = axd18O.contourf(df_p_ave_d18O[locDict[k]].columns, -df_p_ave_d18O[locDict[k]].index, 
                           df_p_ave_d18O[locDict[k]].values, cmap = 'Blues',
                           vmin = d18Omin, vmax = d18Omax)
    if locDict[k] != 2020:
        axd18O.set(xlim = [d1,d2])
    else:
        axd18O.set(xlim = [d1,d22])


    # if locDict[k] == 2019:

    #     cbar = fig.colorbar(
    #         ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
    #         ticks=range(d18Omin, d18Omax+5, 5))
    #     cbar.set_ticks(np.arange(d18Omin,d18Omax,5))
        
    # cbar = fig.colorbar(
    #     ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
    #     ticks=range(d18Omin, d18Omax+5, 5),orientation = 'horizontal')
    # cbar.set_ticks(np.arange(d18Omin,d18Omax,5))


    if locDict[k] == 2017:
        plt.ylabel('Depth (cm)')
        plt.text(datesLeft[1],-40,d18Osym)

    if locDict[k] != 2017:
        axd18O.set_yticklabels([])

    axd18O.set_xticklabels([])
    axd18O.set(ylim = [-50, 20])

    for date in datesLeft:
        plt.text(date,-20,'|')
        plt.text(date-timedelta(days=2.5),-18,'^')
    axd18O.xaxis.label.set_color(dictXfntClr[locDict[k]])

    xtcks = axd18O.get_xticks()
    axd18O.set_xticks(xtcks[0::2])

# dxs contour plots

for k in locDict:
    axdxs = plt.subplot2grid(shape=(5, 6), loc=(3, k), colspan=2, rowspan = 2)
    d1 = pd.to_datetime(str(locDict[k])+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(locDict[k])+'0815',format = '%Y%m%d')
    d22 = pd.to_datetime(str(locDict[k])+'0725',format = '%Y%m%d')

    datesLeft = df_p_ave_dxs[locDict[k]].columns

    cntr = axdxs.contourf(df_p_ave_dxs[locDict[k]].columns, -df_p_ave_dxs[locDict[k]].index, 
                           df_p_ave_dxs[locDict[k]].values, cmap = 'bwr',
                           vmin = dxsmin, vmax = dxsmax)



    # if locDict[k] == 2019:
    #     cbar = fig.colorbar(
    #         ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
    #         ticks=range(dxsmin, dxsmax+5, 5))
    #     cbar.set_ticks(np.arange(dxsmin,dxsmax,5))


    # cbar = fig.colorbar(
    #     ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
    #     ticks=range(dxsmin, dxsmax+5, 5), orientation = 'horizontal')
    # cbar.set_ticks(np.arange(dxsmin,dxsmax,5))

    if locDict[k] == 2017:
        plt.ylabel('Depth (cm)')
        plt.text(datesLeft[1],-40,'d-excess ($^{o}/_{oo}$)')

    if locDict[k] != 2017:
        axdxs.set_yticklabels([])


    plt.xticks(rotation = 25)
    for date in datesLeft:
        plt.text(date,-20,'|')
        plt.text(date-timedelta(days=2.5),-18,'^')

    if locDict[k] != 2020:
        axdxs.set(xlim = [d1,d2])
    else:
        axdxs.set(xlim = [d1,d22])

    axdxs.tick_params(axis='x', colors = dictXfntClr[locDict[k]])

    axdxs.set(ylim = [-50, 20])
    xtcks = axdxs.get_xticks()

    axdxs.set_xticks(xtcks[0::2])


fm.saveas(fig,figureLoc+'T2m_d18O_dxs_aContourSummer_2017_2018_2019')

#############
# save the profiles just produced
#############
meanDataFileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/meanTimeProfiles/'
df_isot_mean_dxs.to_pickle(meanDataFileLoc+dxsMeanTimeProfileFileName)
df_isot_std_dxs.to_pickle(meanDataFileLoc+dxsStdTimeProfileFileName)
# df_isot_min_dxs.to_pickle(meanDataFileLoc+dxsMinTimeProfileFileName)
# df_isot_max_dxs.to_pickle(meanDataFileLoc+dxsMaxTimeProfileFileName)

df_isot_mean_d18O.to_pickle(meanDataFileLoc+d18OMeanTimeProfileFileName)
df_isot_std_d18O.to_pickle(meanDataFileLoc+d18OStdTimeProfileFileName)
# df_isot_min_d18O.to_pickle(meanDataFileLoc+d18OMinTimeProfileFileName)
# df_isot_max_d18O.to_pickle(meanDataFileLoc+d18OMaxTimeProfileFileName)



#############
# d18O vs dxs as a function of various filters
#############

cols = ['d18O','dexcess','depth']
df_isot_sca = df_isot[cols]
# drop nan rows to make things faster
df_isot_sca = df_isot_sca.dropna(thresh = 2)

ageDepthVals = df_isot_sca.index.get_level_values('ageDepth')
coreIDsVals = df_isot_sca.index.get_level_values('coreID')
coreDateVals = df_isot_sca.index.get_level_values('date')

advWinInd =  np.where((ageDepthVals.month == 12)| (ageDepthVals.month == 1)| (ageDepthVals.month == 2))
advWinInd = advWinInd[0]

advSumInd =  np.where((ageDepthVals.month == 6)| (ageDepthVals.month == 7)| (ageDepthVals.month == 8))
advSumInd = advSumInd[0]


fig = plt.figure()
plt.plot(df_isot_sca.d18O,df_isot_sca.dexcess,'.',markersize = 0.5, color = 'black',alpha = 0.1)
plt.plot(df_isot_sca.loc[(ageDepthVals[advWinInd],coreIDsVals,coreDateVals)].d18O,
                 df_isot_sca.loc[(ageDepthVals[advWinInd],coreIDsVals,coreDateVals)].dexcess,
                 '.',color = 'blue',alpha = 0.2,markersize = 0.5)
plt.plot(df_isot_sca.loc[(ageDepthVals[advSumInd],coreIDsVals,coreDateVals)].d18O,
                 df_isot_sca.loc[(ageDepthVals[advSumInd],coreIDsVals,coreDateVals)].dexcess,
                 '.',color = 'red',alpha = 0.2,markersize = 0.5)
plt.xlabel(d18Osym)
plt.ylabel('dxs (per mil)')
plt.xlim([d18Omin,-20])
plt.ylim([-10,30])
fm.saveas(fig, figureLoc+'d18O_vs_dxs_2017-2019')

#############
# d18O and dxs residual profiles
#############

ind = np.arange(0,730,1)
resid_d18O = pd.DataFrame(index = ind,columns = ['2018_2017','2019_2017','2019_2018'])
resid_dxs  = pd.DataFrame(index = ind,columns = ['2018_2017','2019_2017','2019_2018'])

sl = 1
# 2019-2018
end = datetime.datetime(2018,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018]
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2019_2018'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_d18O['2019_2018'] = resid_d18O['2019_2018'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
d18O_diff_temp = eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019])+eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018])
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2019_2018_se'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2018-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&
                                   (df_isot_mean_d18O.index<end)][2018]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017]
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2018_2017'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_d18O['2018_2017'] = resid_d18O['2018_2017'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
d18O_diff_temp = eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018])+eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017])
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2018_2017_se'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2019-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017]
resid_d18O.loc[365:(365+len(d18O_diff_temp))-1,'2019_2017'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_d18O['2019_2017'] = resid_d18O['2019_2017'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
d18O_diff_temp = eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019])+eg.seComp(sl, df_isot_std_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017], df_isot_num_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017])
resid_d18O.loc[365:(365+len(d18O_diff_temp))-1,'2019_2017_se'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

fig = plt.subplots()
plt.subplot(211)
plt.plot(resid_d18O.index/365,resid_d18O['2018_2017'],label = '2018-2017')
plt.plot(resid_d18O.index/365,resid_d18O['2019_2017'],label = '2019-2017')
plt.plot(resid_d18O.index/365,resid_d18O['2019_2018'],label = '2019-2018')
plt.fill_between(resid_d18O.index/365, resid_d18O['2018_2017']+resid_d18O['2018_2017_se'], 
                 np.asarray(resid_d18O['2018_2017']-resid_d18O['2018_2017_se']),color = 'Blue',alpha = 0.2)
plt.fill_between(resid_d18O.index/365, resid_d18O['2019_2017']+resid_d18O['2019_2017_se'], 
                 resid_d18O['2019_2017']-resid_d18O['2019_2017_se'],color = 'Orange',alpha = 0.2)
plt.fill_between(resid_d18O.index/365, resid_d18O['2019_2018']+resid_d18O['2019_2018_se'], 
                 resid_d18O['2019_2018']-resid_d18O['2019_2018_se'],color = 'Green',alpha = 0.2)
plt.plot([1, 1],[-12,12],'--',color = 'Grey')
plt.plot([0, 0],[-12,12],'--',color = 'Grey',label = '31 July')
plt.plot([-0.1,2],[0,0],':',color = 'Grey')
plt.ylabel('$\Delta$' + d18Osym)
plt.ylim([-12,12])
plt.yticks([-10,-5,0,5,10])
plt.xlim([-0.1,2])
plt.xticks(alpha = 0)




# dxs
#2019-2018
end = datetime.datetime(2018,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2019]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2018]
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2019_2018'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_dxs['2019_2018'] = resid_dxs['2019_2018'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
dxs_diff_temp = eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2019], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2019])+eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2018], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2018])
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2019_2018_se'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

# 2018-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2018]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2017]
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2018_2017'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_dxs['2018_2017'] = resid_dxs['2018_2017'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
dxs_diff_temp = eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2018], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2018])+eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2017], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2017])
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2018_2017_se'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

# 2019-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2019]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2017]
resid_dxs.loc[365:(365+len(dxs_diff_temp))-1,'2019_2017'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
resid_dxs['2019_2017'] = resid_dxs['2019_2017'].apply(float)        # fixing mysterious datatype error (objects appears where float should)
dxs_diff_temp = eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2019], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2019])+eg.seComp(sl, df_isot_std_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2017], df_isot_num_dxs[(df_isot_std_dxs.index>=beg)&(df_isot_std_dxs.index<end)][2017])
resid_dxs.loc[365:(365+len(dxs_diff_temp))-1,'2019_2017_se'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

plt.subplot(212)
plt.plot(resid_dxs.index/365,resid_dxs['2018_2017'],label = '2018-2017')
plt.plot(resid_dxs.index/365,resid_dxs['2019_2017'],label = '2019-2017')
plt.plot(resid_dxs.index/365,resid_dxs['2019_2018'],label = '2019-2018')
plt.fill_between(resid_dxs.index/365, resid_dxs['2018_2017']+resid_dxs['2018_2017_se'], 
                 np.asarray(resid_dxs['2018_2017']-resid_dxs['2018_2017_se']),color = 'Blue',alpha = 0.2)
plt.fill_between(resid_dxs.index/365, resid_dxs['2019_2017']+resid_dxs['2019_2017_se'], 
                 resid_dxs['2019_2017']-resid_dxs['2019_2017_se'],color = 'Orange',alpha = 0.2)
plt.fill_between(resid_dxs.index/365, resid_dxs['2019_2018']+resid_dxs['2019_2018_se'], 
                 resid_dxs['2019_2018']-resid_dxs['2019_2018_se'],color = 'Green',alpha = 0.2)
plt.plot([1, 1],[-12,12],'--',color = 'Grey')
plt.plot([0, 0],[-12,12],'--',color = 'Grey',label = '31 July')
plt.plot([-0.1,2],[0,0],':',color = 'Grey')
plt.ylabel('$\Delta$ d-excess ($^{o}/_{oo}$)')
plt.ylim([-12,12])
plt.yticks([-10,-5,0,5,10])
plt.xlim([-0.1,2])
plt.legend(loc = 'upper right', fontsize = 7.5)
plt.xlabel('age of reference profile (years)')
fm.saveas(fig[0], figureLoc+'d18O_dxs_resid_'+str(sl)+'s_yearInSnow')

# answers for mean annual change of d18O and dxs after 1 year in the snow
print('mean annual change of d18O from 2019-2018: ' + str(np.round(np.mean(resid_d18O.loc[np.arange(0,365,1),'2019_2018']),2)))
print('mean annual change of d18O from 2018-2017: ' + str(np.round(np.mean(resid_d18O.loc[np.arange(0,365,1),'2018_2017']),2)))
print('mean annual change of dxs from 2019-2018: ' + str(np.round(np.mean(resid_dxs.loc[np.arange(0,365,1),'2019_2018']),2)))
print('mean annual change of dxs from 2018-2017: ' + str(np.round(np.mean(resid_dxs.loc[np.arange(0,365,1),'2018_2017']),2)))

for i in x: 
    if ~(np.isfinite(i)): 
        print(i); 
    else: 
        print('infinite')
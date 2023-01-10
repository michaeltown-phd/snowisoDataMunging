#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:51:06 2022

EastGRIP isotope data publication plots

@author: michaeltown
"""

# libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse, Polygon
import figureMagic as fm
import EastGRIPprocs as eg
import calendar
import datetime as datetime


#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'per mil' # '\textperthousand'


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


# plot figures by year

dvals = df_isod.index.get_level_values(0).unique()
coreIDs = df_isod.index.get_level_values(1).unique()
dates = df_isod.index.get_level_values(2).unique()

years = dates.year.unique()

coreIDsAll = df_isod.index.get_level_values(1)
yearsAll = df_isod.index.get_level_values(2).year

############
# mean profiles by depth, d18O, dxs and by year
############
for y in years:
    #d18O
    df_isod_mean = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].d18O.groupby(level = 'depth').apply(np.mean)
    df_isod_std = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].d18O.groupby(level = 'depth').apply(np.std)
    df_isod_min = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].d18O.groupby(level = 'depth').apply(np.min)
    df_isod_max = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].d18O.groupby(level = 'depth').apply(np.max)
    
    fig = plt.figure()
    plt.plot(df_isod_mean,-dvals,color = 'Blue',linewidth = 3)
    plt.fill_betweenx(-dvals,df_isod_min,df_isod_max,color = 'Blue',alpha = 0.5)
    plt.plot(df_isod_mean-df_isod_std,-dvals,color = 'Blue',linewidth = 1,linestyle = '--')
    plt.plot(df_isod_mean+df_isod_std,-dvals,color = 'Blue',linewidth = 1,linestyle = '--')
    plt.title(str(y))
    plt.xlabel('d18O (per mil)')
    plt.ylabel('depth (cm)')
    plt.xlim([-50,-20])
    plt.ylim([-100,20])
    fm.saveas(fig, figureLoc+'d18O_depthProfile_'+str(y))
    
    #dxs
    df_isod_mean = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].dexcess.groupby(level = 'depth').apply(np.mean)
    df_isod_std = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].dexcess.groupby(level = 'depth').apply(np.std)
    df_isod_min = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].dexcess.groupby(level = 'depth').apply(np.min)
    df_isod_max = df_isod.loc[(dvals,coreIDsAll,yearsAll == y)].dexcess.groupby(level = 'depth').apply(np.max)
    
    fig = plt.figure()
    plt.plot(df_isod_mean,-dvals,color = 'black',linewidth = 3)
    plt.fill_betweenx(-dvals,df_isod_min,df_isod_max,color = 'black',alpha = 0.5)
    plt.plot(df_isod_mean-df_isod_std,-dvals,color = 'black',linewidth = 1,linestyle = '--')
    plt.plot(df_isod_mean+df_isod_std,-dvals,color = 'black',linewidth = 1,linestyle = '--')
    plt.title(str(y))
    plt.xlabel('dxs (per mil)')
    plt.ylabel('depth (cm)')
    plt.xlim([-10,30])
    plt.ylim([-100,20])
    fm.saveas(fig, figureLoc+'dxs_depthProfile_'+str(y))

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
lineclr_dxs = dict(zip([2017,2018,2019],['DarkGrey','Grey','Black']))
depthVals = df_isod.index.get_level_values(0).unique()
depthValsAll = df_isod.index.get_level_values(0)
coreIDsAlld = df_isod.index.get_level_values(1)
yearsAlld = df_isod.index.get_level_values(2)

# d18O stacked profile
fig = plt.figure()
fig = plt.subplots()
plt.subplot(1,6,(1,2))
for y in years:
    if y == 2017:
        depthValsP = depthVals + acc2017+ acc2018 
    elif y == 2018:
        depthValsP = depthVals + acc2018
    else:
        depthValsP = depthVals

    df_isod_mean[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.mean)
    df_isod_std[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.std)
    df_isod_min[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.min)
    df_isod_max[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').d18O.apply(np.max)

    # plot the profile data
    plt.plot(df_isod_mean[y],-depthValsP,color = lineclr_d18O[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
#    plt.fill_betweenx(-depthValsP,df_isod_min[y],df_isod_max[y],color = 'Blue',alpha = 0.2,
                      hatch = hatchDict[y])
    plt.fill_betweenx(-depthValsP,df_isod_mean[y]-df_isod_std[y],df_isod_mean[y]+df_isod_std[y],color = 'Blue',alpha = 0.5)
#    plt.plot(df_isod_mean[y]-df_isod_std[y],-depthValsP,color = lineclr_d18O[y],linewidth = 1,linestyle = '--')
#    plt.plot(df_isod_mean[y]+df_isod_std[y],-depthValsP,color = lineclr_d18O[y],linewidth = 1,linestyle = '--')

plt.xlabel(d18Osym)
plt.xlim([-50,-20])
plt.ylim([-200,20])
for y in np.arange(-160,20,40):
    plt.plot([-50,-20],[y,y],':',color = 'Grey')

plt.yticks(np.arange(-200,40,20))
plt.legend(loc = 'upper left')

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,6,3)
prof2017to2019 = pd.DataFrame(df_isod_mean[2017].copy())
prof2018to2019 = pd.DataFrame(df_isod_mean[2018].copy())
prof2019 = pd.DataFrame(df_isod_mean[2019].copy())

prof2017to2019['newDepth'] = depthVals+acc2017+acc2018
prof2018to2019['newDepth'] = depthVals+acc2018

prof2017to2019 = prof2017to2019.set_index('newDepth')
prof2018to2019 = prof2018to2019.set_index('newDepth')
nDepthVals2018 = prof2018to2019.index.values
nDepthVals2017 = prof2017to2019.index.values

dfMeanDepth_d18O = pd.DataFrame(index = np.arange(-20,191,1),columns = [2017,2018,2019])

for i in prof2017to2019.index:
    dfMeanDepth_d18O.loc[i,2017] = prof2017to2019.loc[i,'d18O']

for i in prof2018to2019.index:
    dfMeanDepth_d18O.loc[i,2018] = prof2018to2019.loc[i,'d18O']

for i in prof2019.index:
    dfMeanDepth_d18O.loc[i,2019] = prof2019.loc[i,'d18O'] 

plt.plot(dfMeanDepth_d18O[2018]-dfMeanDepth_d18O[2017],-dfMeanDepth_d18O.index, label = '2018-2017')
plt.plot(dfMeanDepth_d18O[2019]-dfMeanDepth_d18O[2017],-dfMeanDepth_d18O.index, label = '2019-2017')
plt.plot(dfMeanDepth_d18O[2019]-dfMeanDepth_d18O[2018],-dfMeanDepth_d18O.index, label = '2019-2018')
plt.xlabel(d18Osym)
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.ylim([-190,20])
plt.yticks(np.arange(-200,40,20),alpha = 0)
for y in np.arange(-160,20,40):
    plt.plot([-12,12],[y,y],':',color = 'Grey')
plt.plot([0,0],[-190,20],':',color = 'black')
plt.legend()
plt.ylabel('depth (cm)')

# fm.saveas(fig[0], figureLoc+'d18O_depthProfileStacked_2017_2019')

# dxs
plt.subplot(1,6,(4,5))
for y in years:
    if y == 2017:
        depthValsP = depthVals + acc2017+ acc2018 
    elif y == 2018:
        depthValsP = depthVals + acc2018
    else:
        depthValsP = depthVals

    df_isod_mean[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.mean)
    df_isod_std[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.std)
    df_isod_min[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.min)
    df_isod_max[y] = df_isod.loc[(depthVals,coreIDsAlld,yearsAlld.year == y)].groupby(level = 'depth').dexcess.apply(np.max)

    # plot the profile data
    plt.plot(df_isod_mean[y],-depthValsP,color = lineclr_dxs[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(-depthValsP,df_isod_min[y],df_isod_max[y],color = 'Grey',alpha = 0.2,
                      hatch = hatchDict[y])
    plt.plot(df_isod_mean[y]-df_isod_std[y],-depthValsP,color = lineclr_dxs[y],linewidth = 1,linestyle = '--')
    plt.plot(df_isod_mean[y]+df_isod_std[y],-depthValsP,color = lineclr_dxs[y],linewidth = 1,linestyle = '--')

plt.xlabel('dxs (per mil)')
plt.xlim([-10,30])
plt.ylim([-200,20])
plt.yticks(np.arange(-200,40,20))
plt.ylabel('depth (cm)')
for y in np.arange(-160,20,40):
    plt.plot([-10,30],[y,y],':',color = 'Grey')

plt.legend(loc = 'upper left')

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,6,6)
prof2017to2019 = pd.DataFrame(df_isod_mean[2017].copy())
prof2018to2019 = pd.DataFrame(df_isod_mean[2018].copy())
prof2019 = pd.DataFrame(df_isod_mean[2019].copy())

prof2017to2019['newDepth'] = depthVals+acc2017+acc2018
prof2018to2019['newDepth'] = depthVals+acc2018

prof2017to2019 = prof2017to2019.set_index('newDepth')
prof2018to2019 = prof2018to2019.set_index('newDepth')
nDepthVals2018 = prof2018to2019.index.values
nDepthVals2017 = prof2017to2019.index.values

dfMeanDepth_dxs = pd.DataFrame(index = np.arange(-20,191,1),columns = [2017,2018,2019])

for i in prof2017to2019.index:
    dfMeanDepth_dxs.loc[i,2017] = prof2017to2019.loc[i,'dexcess']

for i in prof2018to2019.index:
    dfMeanDepth_dxs.loc[i,2018] = prof2018to2019.loc[i,'dexcess']

for i in prof2019.index:
    dfMeanDepth_dxs.loc[i,2019] = prof2019.loc[i,'dexcess'] 

plt.plot(dfMeanDepth_dxs[2018]-dfMeanDepth_dxs[2017],-dfMeanDepth_dxs.index, label = '2018-2017')
plt.plot(dfMeanDepth_dxs[2019]-dfMeanDepth_dxs[2017],-dfMeanDepth_dxs.index, label = '2019-2017')
plt.plot(dfMeanDepth_dxs[2019]-dfMeanDepth_dxs[2018],-dfMeanDepth_dxs.index, label = '2019-2018')
plt.xlabel('dxs (per mil)')
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.ylim([-190,20])
plt.yticks(np.arange(-200,40,20),alpha = 0)
plt.plot([0,0],[-200,20],':',color = 'black')
plt.legend()
for y in np.arange(-160,20,40):
    plt.plot([-12,12],[y,y],':',color = 'Grey')
fm.saveas(fig[0], figureLoc+'d18Odxs_depthProfileStacked_2017_2019_'+vers)




############
# mean contour time/depth series
############
#d18O
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




    # set date range for the plot
    d1 = pd.to_datetime(str(y)+'0501',format = '%Y%m%d')
    d2 = pd.to_datetime(str(y)+'0815',format = '%Y%m%d')
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'Blues',vmin = -50, vmax = -20)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')    
    ax2[1].set(xlim = [d1,d2])
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-50, -20+5, 5))
    cbar.set_ticks(np.arange(-50,-20,5))
    plt.ylim(-50,10)
    plt.ylabel('depth (cm)')
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
    plt.ylim(-50,10)
    plt.ylabel('depth (cm)')
    plt.xticks(rotation = 25)
    for date in datesLeft:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(datesLeft[1],-40,'std d18O')
    fm.saveas(fig1, figureLoc+'d18O_depthTimeContour_std_'+str(y))

#dxs
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
    plt.ylim(-50,5)
    plt.xlabel('date')
    plt.ylabel('depth (cm)')
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
    plt.ylim(-50,10)
    plt.ylabel('depth (cm)')
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
        cntr = ax2[1].contourf(df_p_ave.columns, -df_p_ave.index, df_p_ave.values, cmap = 'Blues',vmin = -50, vmax = -20)
        ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
        ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
        ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
        ax2[0].set_ylabel('T (oC)')
        ax2[0].set(xlim = [d1,d2])
        ax2[0].set_xticklabels('')
        ax2[1].set(xlim = [d1,d2])
        cbar = fig1.colorbar(
            ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
            ticks=range(-50, -20+5, 5))
        cbar.set_ticks(np.arange(-50,-20,5))
        plt.ylim(-50,10)
        plt.ylabel('depth (cm)')
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
            ticks=range(-50, -20+5, 5))
        cbar.set_ticks(np.arange(-10,25,5))
        plt.ylim(-50,10)
        plt.ylabel('depth (cm)')
        plt.xticks(rotation = 25)
        for date in datesLeft:
            plt.text(date,-20,' |')
            plt.text(date,-18,'^')
        plt.text(datesLeft[1],-40,'dxs for p ' + str(c))
        fm.saveas(fig1, figureLoc+'dxs_depthTimeContour_'+str(y)+'_p_'+ str(c))

'''
#############
# mean profiles of d18O as a function of time
#############

lnstyDict = dict(zip([2017,2018,2019],['-','-','-']))
hatchDict = dict(zip([2017,2018,2019],['///','\\/','']))
lineclr = dict(zip([2017,2018,2019],['SlateBlue','Blue','DarkBlue']))


timeValsAll = df_isot.index.get_level_values(0)
timeVals = df_isot.index.get_level_values(0).unique()
coreIDsAllt = df_isot.index.get_level_values(1)
yearsAllt = df_isot.index.get_level_values(2)
df_isot_mean_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isot_std_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isot_min_d18O = pd.DataFrame(columns = [2017,2018,2019])
df_isot_max_d18O = pd.DataFrame(columns = [2017,2018,2019])

fig = plt.subplots()
plt.subplot(1,3,(1,2))
for y in years:
    df_isot_mean_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.mean)
    df_isot_std_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.std)
    df_isot_min_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.min)
    df_isot_max_d18O[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').d18O.apply(np.max)

    # plot the profile data
    plt.plot(df_isot_mean_d18O[y],timeVals,color = lineclr[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(timeVals,df_isot_min_d18O[y],df_isot_max_d18O[y],color = 'Blue',alpha = 0.2,
                      hatch = hatchDict[y])
    plt.plot(df_isot_mean_d18O[y]-df_isot_std_d18O[y],timeVals,color = 'Blue',linewidth = 1,linestyle = '--')
    plt.plot(df_isot_mean_d18O[y]+df_isot_std_d18O[y],timeVals,color = 'Blue',linewidth = 1,linestyle = '--')
    plt.xlabel(d18Osym)
    plt.xlim([-50,-20])
    plt.ylim([pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)])

plt.legend(loc = 'upper left')
# set horizontal lines at summer peaks
yline = np.asarray([])
for i in np.arange(2015,2021,1):
    yline = np.append(yline,pd.to_datetime(str(i)+'0731',format = '%Y%m%d'))

for y in yline:
    plt.plot([-50,-20],[y,y],':',color = 'Grey')

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,3,3)
plt.plot(df_isot_mean_d18O[2018]-df_isot_mean_d18O[2017],timeVals, label = '2018-2017')
plt.plot(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2017],timeVals, label = '2019-2017')
plt.plot(df_isot_mean_d18O[2019]-df_isot_mean_d18O[2018],timeVals, label = '2019-2018')
plt.xlabel(d18Osym)
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.yticks(color = 'White',alpha = 0)
plt.ylim([pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)])
plt.legend()
plt.plot([0,0],[pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)],'--',linewidth = 0.25,color = 'Black')
for y in yline:
    plt.plot([-12,12],[y,y],':',color = 'Grey')

fm.saveas(fig[0], figureLoc+'d18O_depthAgeProfile_2017_2019')

###########
# mean profiles of dxs as a function of time
###########

lnstyDict = dict(zip([2017,2018,2019],['-','-','-']))
hatchDict = dict(zip([2017,2018,2019],['///','\\/','']))
lineclr = dict(zip([2017,2018,2019],['DarkGrey','Grey','Black']))

timeValsAll = df_isot.index.get_level_values(0)
timeVals = df_isot.index.get_level_values(0).unique()
coreIDsAllt = df_isot.index.get_level_values(1)
yearsAllt = df_isot.index.get_level_values(2)
df_isot_mean_dxs = pd.DataFrame(columns = [2017,2018,2019])
df_isot_std_dxs = pd.DataFrame(columns = [2017,2018,2019])
df_isot_min_dxs = pd.DataFrame(columns = [2017,2018,2019])
df_isot_max_dxs = pd.DataFrame(columns = [2017,2018,2019])

fig1 = plt.subplots()
plt.subplot(1,3,(1,2))
for y in years:
    df_isot_mean_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.mean)
    df_isot_std_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.std)
    df_isot_min_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.min)
    df_isot_max_dxs[y] = df_isot.loc[(timeValsAll,coreIDsAllt,yearsAllt.year == y)].groupby(level = 'ageDepth').dexcess.apply(np.max)

    # plot the profile data
    plt.plot(df_isot_mean_dxs[y],timeVals,color = lineclr[y],linewidth = 3, linestyle = lnstyDict[y],label = str(y))
    plt.fill_betweenx(timeVals,df_isot_min_dxs[y],df_isot_max_dxs[y],color = 'Grey',alpha = 0.2,
                      hatch = hatchDict[y])
    plt.plot(df_isot_mean_dxs[y]-df_isot_std_dxs[y],timeVals,color = 'Grey',linewidth = 1,linestyle = '--')
    plt.plot(df_isot_mean_dxs[y]+df_isot_std_dxs[y],timeVals,color = 'Grey',linewidth = 1,linestyle = '--')
    plt.xlabel('dxs (per mil)')
    plt.xlim([-10,30])
    plt.ylim([pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)])

plt.legend(loc = 'upper right')
# set horizontal lines at summer peaks
yline = np.asarray([])
for i in np.arange(2015,2021,1):
    yline = np.append(yline,pd.to_datetime(str(i)+'0731',format = '%Y%m%d'))

for y in yline:
    plt.plot([-10,30],[y,y],':',color = 'Grey')

# plot the residual profile data in an adjacent subplot 
plt.subplot(1,3,3)
plt.plot(df_isot_mean_dxs[2018]-df_isot_mean_dxs[2017],timeVals, label = '2018-2017')
plt.plot(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2017],timeVals, label = '2019-2017')
plt.plot(df_isot_mean_dxs[2019]-df_isot_mean_dxs[2018],timeVals, label = '2019-2018')
plt.xlabel('dxs (per mil)')
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.yticks(color = 'White',alpha = 0)
plt.ylim([pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)])
plt.legend()
plt.plot([0,0],[pd.to_datetime('20150101',format = '%Y%m%d'),np.max(timeVals)],'--',linewidth = 0.25,color = 'Black')
for y in yline:
    plt.plot([-12,12],[y,y],':',color = 'Grey')

fm.saveas(fig1[0], figureLoc+'dxs_depthAgeProfile_2017-2019')

#############
# save the profiles just produced
#############
meanDataFileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/meanTimeProfile/'
df_isot_mean_dxs.to_pickle(meanDataFileLoc+dxsMeanTimeProfileFileName)
df_isot_std_dxs.to_pickle(meanDataFileLoc+dxsStdTimeProfileFileName)
df_isot_min_dxs.to_pickle(meanDataFileLoc+dxsMinTimeProfileFileName)
df_isot_max_dxs.to_pickle(meanDataFileLoc+dxsMaxTimeProfileFileName)

df_isot_mean_d18O.to_pickle(meanDataFileLoc+d18OMeanTimeProfileFileName)
df_isot_std_d18O.to_pickle(meanDataFileLoc+d18OStdTimeProfileFileName)
df_isot_min_d18O.to_pickle(meanDataFileLoc+d18OMinTimeProfileFileName)
df_isot_max_d18O.to_pickle(meanDataFileLoc+d18OMaxTimeProfileFileName)

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
plt.xlim([-50,-20])
plt.ylim([-10,30])
fm.saveas(fig, figureLoc+'d18O_vs_dxs_2017-2019')

#############
# d18O and dxs residual profiles
#############

ind = np.arange(0,730,1)
resid_d18O = pd.DataFrame(index = ind,columns = ['2018_2017','2019_2017','2019_2018'])
resid_dxs  = pd.DataFrame(index = ind,columns = ['2018_2017','2019_2017','2019_2018'])


end = datetime.datetime(2018,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018]
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2019_2018'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2018-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2018]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017]
resid_d18O.loc[0:len(d18O_diff_temp)-1,'2018_2017'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2019-2018
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
d18O_diff_temp = df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2019]-df_isot_mean_d18O[(df_isot_mean_d18O.index>=beg)&(df_isot_mean_d18O.index<end)][2017]
resid_d18O.loc[365:(365+len(d18O_diff_temp))-1,'2019_2017'] = d18O_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

fig = plt.subplots()
plt.subplot(121)
plt.plot(resid_d18O['2018_2017'],resid_d18O.index/365,label = '2018-2017')
plt.plot(resid_d18O['2019_2017'],resid_d18O.index/365,label = '2019-2017')
plt.plot(resid_d18O['2019_2018'],resid_d18O.index/365,label = '2019-2018')
plt.plot([-12,12],[1, 1],'--',color = 'Grey')
plt.plot([-12,12],[0, 0],'--',color = 'Grey',label = '31 July')
plt.plot([0,0],[-10,730],':',color = 'Grey')
plt.xlabel('resid ' + d18Osym)
plt.ylabel('years in snow')
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.ylim([2,-0.1])
plt.legend(loc = 'lower right', fontsize = 7.5)


# dxs
end = datetime.datetime(2018,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2019]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2018]
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2019_2018'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2018-2017
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2018]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2017]
resid_dxs.loc[0:len(dxs_diff_temp)-1,'2018_2017'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top
# 2019-2018
end = datetime.datetime(2017,7,31)
beg = datetime.datetime(2016,7,31)
dxs_diff_temp = df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2019]-df_isot_mean_dxs[(df_isot_mean_dxs.index>=beg)&(df_isot_mean_dxs.index<end)][2017]
resid_dxs.loc[365:(365+len(dxs_diff_temp))-1,'2019_2017'] = dxs_diff_temp.values[::-1]       # need to reverse these for the newest to be on top

plt.subplot(122)
plt.plot(resid_dxs['2018_2017'],resid_dxs.index/365,label = '2018-2017')
plt.plot(resid_dxs['2019_2017'],resid_dxs.index/365,label = '2019-2017')
plt.plot(resid_dxs['2019_2018'],resid_dxs.index/365,label = '2019-2018')
plt.plot([-12,12],[1, 1],'--',color = 'Grey')
plt.plot([-12,12],[0, 0],'--',color = 'Grey',label = '31 July')
plt.plot([0,0],[-10,730],':',color = 'Grey')
plt.xlabel('resid dxs')
plt.xlim([-12,12])
plt.xticks([-10,-5,0,5,10])
plt.ylim([2,-0.1])
plt.yticks(alpha = 0)
fm.saveas(fig[0], figureLoc+'d18O_dxs_resid_yearInSnow')

# answers for mean annual change of d18O and dxs after 1 year in the snow
print('mean annual change of d18O from 2019-2018: ' + str(np.round(np.mean(resid_d18O.loc[np.arange(0,365,1),'2019_2018']),2)))
print('mean annual change of d18O from 2018-2017: ' + str(np.round(np.mean(resid_d18O.loc[np.arange(0,365,1),'2018_2017']),2)))
print('mean annual change of dxs from 2019-2018: ' + str(np.round(np.mean(resid_dxs.loc[np.arange(0,365,1),'2019_2018']),2)))
print('mean annual change of dxs from 2018-2017: ' + str(np.round(np.mean(resid_dxs.loc[np.arange(0,365,1),'2018_2017']),2)))
'''

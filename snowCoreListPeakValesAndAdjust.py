#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:17:58 2022

List the snow core individual profile peak locations (depths) and sample numbers
allows user to adjust as necessary

Then this script will classify the sections of the profile based on how many features have been encountered.
The goal will be to have each profile have the same number of features... Not always possible, but we will shoe-horn
this one...

@author: michaeltown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from scipy.signal import find_peaks
import figureMagic as fm
from matplotlib.cm import ScalarMappable
import datetime as dt

# main

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/2019/'

fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'
fileNameAddPeaks = 'addPeaksEastGRIPsnowiso.csv'
fileNameSubtractPeaks = 'subtractPeaksEastGRIPsnowiso.csv'

df_iso = pd.read_pickle(fileLoc+fileNameIso);

df_iso_temp = df_iso[df_iso.peaks == 1].depthAcc_reg;

os.chdir(fileLoc)

df = df_iso_temp.sort_index(ascending = True)

df.to_csv(r'./snowCorePeakDepths.txt',header=True,index=True,sep = ' ')

# focusing on 2019

df_iso_temp = df_iso[(df_iso.peaks == 1)&(df_iso.year == 2019)].depthAcc_reg;
df = df_iso_temp.sort_index(ascending = True)
df.to_csv(r'./snowCorePeakDepths2019.txt',header=True,index=True,sep = ' ')

# insert these data to position 1
#addListNames = ['SP1_20190611', 'SP1_20190626', 'SP1_20190715', 'SP1_20190724']
#addListDepths = [18.1, 29.1, 31.3, 26.9]
df_subtractPeaks = pd.read_csv(fileLoc+fileNameSubtractPeaks);
df_addPeaks = pd.read_csv(fileLoc+fileNameAddPeaks);


# start here to add the next positions across 2019 



# this loop adds the new peaks to the data set
for i in df_addPeaks.index:
#for k in addDict.keys():
    
    k = df_addPeaks[df_addPeaks.index == i].sampleName.values[0];
    p = int(k[2:3])
    y = int(k[4:8])
    m = int(k[8:10])
    d = int(k[10:12])
    depth = df_addPeaks.loc[df_addPeaks.index == i,'addPeakLocation'].values[0]
    peakAdd = df_iso[(df_iso.year == y) & (df_iso.month == m)&(df_iso.day == d)&
                     (df_iso.depthAcc_reg == depth)&(df_iso.coreID == p)].peaks
    df_iso.loc[peakAdd.index,'peaks'] = 1



# remove these data


# this loop will take away peaks
for i in df_subtractPeaks.index:
#for k in subtractDict.keys():

    k = df_subtractPeaks[df_subtractPeaks.index == i].sampleName.values[0];
    p = int(k[2:3])
    y = int(k[4:8])
    m = int(k[8:10])
    d = int(k[10:12])
    depth = df_subtractPeaks.loc[df_subtractPeaks.index == i,'subtractPeakLocation'].values[0]
    peakSub = df_iso[(df_iso.year == y) & (df_iso.month == m)&(df_iso.day == d)&
                     (df_iso.depthAcc_reg == depth)&(df_iso.coreID == p)].peaks
    df_iso.loc[peakSub.index,'peaks'] = 0

# check the peak adjustment
coreID = np.arange(1,6);
yearUnique = df_iso.year.unique();

df_iso['block'] = 0;
df_iso['timeScale'] = np.nan;

rows = np.arange(0,7) # looks like 6 events in 2019 data set


# time scale for 2019
tsDict2019 = dict(zip(rows,pd.to_datetime(['2019-07-24','2019-03-15','2018-11-1','2018-08-01',
                                           '2018-02-01','2017-08-01', '2017-04-01'])))

for y in yearUnique[-1:]:
    
    
    for c in coreID:  
        dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)]    
        

        figO18 = plt.figure()        
        dateUnique = pd.to_datetime(dfTemp.date.unique());
        numDates = len(dateUnique)
        i = 1;
        for d in dateUnique:
            
            iso18O = dfTemp[(dfTemp.date == d)].d18O
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar
            maxMin = dfTemp[(dfTemp.date == d)&(dfTemp.peaks == 1 )].index
            peaks = dfTemp[(dfTemp.date == d)].peaks
            
            iso18O.sort_index(ascending = True, inplace=True)
            depth.sort_index(ascending = True, inplace=True)
            brksTemp.sort_index(ascending = True, inplace=True)
            hrsTemp.sort_index(ascending = True, inplace=True)
            peaks.sort_index(ascending = True, inplace=True)
            
            block = peaks.cumsum()
            block[len(block)-1] = block[len(block)-1]+1;
            
            # load block values into df, assign timeScale values to initial block times
            
            count = 0;
            
            for b in block.index:
                bVal = block[block.index == b].values
                df_iso.loc[b,'block'] = bVal
                
                if count == bVal:     # should execute once per block
                    df_iso.loc[b,'timeScale'] = tsDict2019[bVal[0]]
                    count += 1
                
                
            
                    
                        
            # take the block data and insert back into the original df with the index, easy short loop.
                        
            if i == 3:
                titleStr = 'individual d18O: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,iso18O,brksTemp*np.nan,hrsTemp*np.nan,-1*depth,titleStr,'d18O','depth (cm)',[-50,-20],[-100,15])
            plt.plot(iso18O[maxMin],-depth[maxMin],'x',color = 'orange')
            i = i + 1;

        plt.show()

# fill in the time scale values for each profile

for y in yearUnique[-1:]:
    
    
    for c in coreID:  
        dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)] 
        dateUnique = pd.to_datetime(dfTemp.date.unique());        

        for d in dateUnique:

            dfDay = dfTemp[dfTemp.date == d];
            dfDayNotNull = dfDay.dropna(subset=['timeScale'])
            dfDayNotNull = dfDayNotNull.sort_index()
            
            # reset row index to numbers to be able to count between the rows
            dfDay['sampleNameFull'] = dfDay.index;
            dfDay = dfDay.sort_index()            
            dfDay.sort_values('sampleNameFull',ascending = True,inplace = True)
            dfDay = dfDay.set_index(np.arange(len(dfDay.timeScale)))

            ind = dfDayNotNull.index;
            
            # should loop around each block once, excluding the last date that brackets the final measurement
            for i in np.arange(len(ind)-1):
                    
                    # finds the time difference between two adjacent preset time values
                    beg =dfDayNotNull.loc[ind[i],'timeScale']
                    end = dfDayNotNull.loc[ind[i+1],'timeScale']
                    begInd = dfDay[dfDay.sampleNameFull == ind[i]].index[0]
                    endInd = dfDay[dfDay.sampleNameFull == ind[i+1]].index[0]
                    timeDelta = end-beg
                    periodNum = endInd - begInd + 1
                    timeRangeTemp = pd.date_range(beg,end,periods=periodNum)
                    dfDay.iloc[begInd:endInd+1,dfDay.columns.get_loc('timeScale')]=timeRangeTemp
            
            # assign dates to df_iso




# build a dataframe with the block stats 
pos = np.arange(1,6) #for later when we've done the other positions
years = 2019; # np.arange(2017,2020) for later when we've done the other years
dates = df_iso[df_iso.year == years].date.unique()

cols = ['block','date','position','d18O','d18O_std','dD','dD_std','dexcess','dexcess_std','dxsln','dxsln_std']
df_blockStats = pd.DataFrame(columns = cols); # can start here with data frame only if doing one year

for d in dates:
    
    for p in pos: #don't need to iterate around position yet

        y = years;
        for r in rows:
            
            d18O = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].d18O.mean()
            d18O_std = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].d18O.std()
            dD = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dD.mean()
            dD_std = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dD.std()
            dexcess = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dexcess.mean()
            dexcess_std = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dexcess.std()
            dxsln = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dxsln.mean()
            dxsln_std = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == r)&(df_iso.year == y)].dxsln.std()
    
            
            values = [r, d, p, d18O, d18O_std, dD, dD_std, dexcess, dexcess_std, dxsln, dxsln_std]
            df_blockStats.loc[len(df_blockStats.index)] = values


# plot all the values and fill across the plot for block values
# chose values in the range of dates in 'dates'
alphas = np.arange(0.2,0.8,0.1) 

# d18O
for d in dates:


    for p in pos:
        fig18O = plt.figure()
        plt.xlim([-50,-20])
        plt.ylim([-100,15])
        plt.grid()

        for a,b in zip(alphas, rows):
            depth = -df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].depthAcc_reg
            d18O = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].d18O
        #    fm.plotProfile3(d18O,depth,clr,d18Oblockvals)
            plt.plot(d18O,depth,'.',color = 'black')
            plt.axhspan(min(depth), max(depth), alpha=a, color='gray', zorder=0)
            blockVal = df_blockStats[(df_blockStats.block == b)&(df_blockStats.date == d)&(df_blockStats.position == p)].d18O.values[0]
            plt.text(-30,np.mean(depth),str(np.round(blockVal)))
        plt.title('eastGRIP pos = ' + str(p) + ', ' + str(d)[:10])
        plt.xlabel('d18O (per mil)')
        plt.ylabel('depth (cm)')
        fig18O.savefig(figureLoc + 'snowCoreEastGRIP_d18O_pos' + str(p) + '_'+ str(d)[:10]+'.jpg')

# dxs
for d in dates:


    for p in pos:
        figdxs = plt.figure()
        plt.xlim([-10,25])
        plt.ylim([-100,15])
        plt.grid()

        for a,b in zip(alphas, rows):
            depth = -df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].depthAcc_reg
            dxs = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].dexcess
        #    fm.plotProfile3(d18O,depth,clr,d18Oblockvals)
            plt.plot(dxs,depth,'.',color = 'blue')
            plt.axhspan(min(depth), max(depth), alpha=a, color='lightblue', zorder=0)
            blockVal = df_blockStats[(df_blockStats.block == b)&(df_blockStats.date == d)&(df_blockStats.position == p)].dexcess.values[0]
            plt.text(20,np.mean(depth),str(np.round(blockVal)))
        plt.title('eastGRIP pos = ' + str(p) + ', ' + str(d)[:10])
        plt.xlabel('dxs (per mil)')
        plt.ylabel('depth (cm)')
        figdxs.savefig(figureLoc + 'snowCoreEastGRIP_dxs_pos' + str(p) + '_'+ str(d)[:10]+'.jpg')


# contour plot of the data to see evolution of the data

# remove duplicate values - these are troublesome and I should go back to remove them altogether.

df_noDups = df_iso.duplicated(['date','coreID','depthAcc_reg'],keep = 'first')
df_test = df_iso[~df_noDups]

# load the PROMICE data, parse for the summer of 2019
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/')
dataFileName = 'eastGRIP_PROMICEData_2016-2019.pkl';
df_promice = pd.read_pickle(dataFileName)
d1 = dates[0]
d2 = dates[4]

dd1 = pd.to_datetime('20190529',format = '%Y%m%d')
dd2 = pd.to_datetime('20190807',format = '%Y%m%d')


# create pivot tables for each of the important data sets I'm interested in, d18O, dexcess

y = 2019;
for p in pos:
    df_d18O_p = df_test[(df_test.year == y)&(df_test.coreID == p)].pivot(index = 'depthAcc_reg', columns = 'date',values= 'd18O')
    df_dxs_p = df_test[(df_test.year == y)&(df_test.coreID == p)].pivot(index = 'depthAcc_reg', columns = 'date',values= 'dexcess')

    cols = df_d18O_p.columns
    numCols = np.arange(len(cols))
    dictCols = dict(zip(cols,numCols))

#    df_d18O_p.rename(columns = dictCols, inplace = True)
#    df_dxs_p.rename(columns = dictCols, inplace = True)

    # plotting the d18O data with Promice data
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_d18O_p.columns, -df_d18O_p.index, df_d18O_p.values, cmap = 'Greys',vmin = -50, vmax = -20)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')    
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-50, -20+5, 5))
    cbar.set_ticks(np.arange(-50,-20,5))
    plt.ylim(-50,5)
    plt.xlabel('date')
    plt.ylabel('depth (cm)')
    plt.xticks(rotation = 25)
    for date in dates:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(dates[1],-40,'d18O, pos = ' + str(p) + ', ' + str(y))
    fig1.savefig(figureLoc + 'snowCoreEastGRIP_d18O_T_pos'+ str(p) + '_contour.jpg')

    # plotting the dxs data with Promice data
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_dxs_p.columns, -df_dxs_p.index, df_dxs_p.values, cmap = 'bwr',vmin = -10, vmax = 25)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-10, 25+5, 5))
    cbar.set_ticks(np.arange(-10,25,5))
    plt.ylim(-50,5)
    plt.xlabel('date')
    plt.ylabel('depth (cm)')
    for date in dates:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(dates[1],-40,'dxs, pos = ' + str(p) + ', ' + str(y))
    plt.xticks(rotation = 25)
    fig1.savefig(figureLoc + 'snowCoreEastGRIP_dxs_T_pos'+ str(p) + '_contour.jpg')

# residual plots of d18O and dxs
y = 2019;
for p in pos:
    df_d18O_p = df_test[(df_test.year == y)&(df_test.coreID == p)].pivot(index = 'depthAcc_reg', columns = 'date',values= 'd18O')
    df_dxs_p = df_test[(df_test.year == y)&(df_test.coreID == p)].pivot(index = 'depthAcc_reg', columns = 'date',values= 'dexcess')

    cols = df_d18O_p.columns
    numCols = np.arange(len(cols))
    dictCols = dict(zip(cols,numCols))

#    df_d18O_p.rename(columns = dictCols, inplace = True)
#    df_dxs_p.rename(columns = dictCols, inplace = True)

    # plotting the d18O data with Promice data
    fig1, ax2  = plt.subplots(2,1);
    cntr = ax2[1].contourf(df_d18O_p.columns, -df_d18O_p.index, df_d18O_p.values-df_d18O_p.values[:,0].reshape(len(df_d18O_p.values[:,0]),1), cmap = 'bwr',vmin = -10, vmax = 10)
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-10, 10, 2))
    cbar.set_ticks(np.arange(-10,10,2))
    plt.ylim(-50,5)
    plt.xlabel('date')
    plt.ylabel('depth (cm)')
    plt.xticks(rotation = 25)
    for date in dates:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(dates[1],-40,'resid d18O, pos = ' + str(p) + ', ' + str(y))
    fig1.savefig(figureLoc + 'snowCoreEastGRIP_d18O_T_pos_resid'+ str(p) + '_contour.jpg')

    # plotting the dxs data with Promice data
    fig1, ax2  = plt.subplots(2,1);
    ax2[1].set(position = [0.1, 0.05, 0.65, 0.4])
    cntr = ax2[1].contourf(df_dxs_p.columns, -df_dxs_p.index, df_dxs_p.values-df_dxs_p.values[:,0].reshape(len(df_dxs_p.values[:,0]),1), cmap = 'bwr',vmin = -10, vmax = 10)
    ax2[0].plot(df_promice[(df_promice.index>d1)&(df_promice.index<d2)].index,df_promice[(df_promice.index>d1)&(df_promice.index<d2)].AirTemperatureC)
    ax2[0].set_ylabel('T (oC)')
    ax2[0].set(position = [0.125, 0.55, 0.62, 0.4])
    ax2[0].set(xlim = [d1,d2])
    ax2[0].set_xticklabels('')
    cbar = fig1.colorbar(
        ScalarMappable(norm=cntr.norm, cmap=cntr.cmap),
        ticks=range(-10, 10, 2))
    cbar.set_ticks(np.arange(-10,10,2))
    plt.ylim(-50,5)
    plt.xlabel('date')
    plt.ylabel('depth (cm)')
    for date in dates:
        plt.text(date,-20,' |')
        plt.text(date,-18,'^')
    plt.text(dates[1],-40,'resid dxs, pos = ' + str(p) + ', ' + str(y))
    plt.xticks(rotation = 25)
    fig1.savefig(figureLoc + 'snowCoreEastGRIP_dxs_T_pos_resid'+ str(p) + '_contour.jpg')


# what was this going to be?

'''
alphas = np.arange(0.2,0.8,0.1) 
for d in dates:


    for p in pos:
        plt.figure()
        plt.xlim([-50,-20])
        #    plt.ylim([-100,15])
        plt.grid()
        
        for a,b in zip(alphas, rows):
            tsPlt = df_iso[(df_iso.coreID == p)&(df_iso.block == b)&(df_iso.date == d)].timeScale
            d18O = df_iso[(df_iso.coreID == p)&(df_iso.block == b)&(df_iso.date == d)].d18O   
        #    fm.plotProfile3(d18O,depth,clr,d18Oblockvals)
            plt.plot(d18O,tsPlt,'.',color = 'black')
            plt.axhspan(min(tsPlt), max(tsPlt), alpha=a, color='gray', zorder=0)
            blockVal = df_blockStats[(df_blockStats.block == b)&(df_blockStats.date == d)&(df_blockStats.position == p)].d18O.values[0]
            plt.text(-30,min(tsPlt)+0.5*(max(tsPlt)-min(tsPlt)),str(np.round(blockVal)))   
        plt.xlim([-50,-20])
        plt.grid()
        plt.title('eastGRIP pos = ' + str(p) + ', ' + str(d))
        plt.xlabel('d18O (per mil)')
        plt.ylabel('depth (cm)')
'''

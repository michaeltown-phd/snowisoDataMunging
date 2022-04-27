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


# main

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);

df_iso_temp = df_iso[df_iso.peaks == 1].depthAcc_reg;

os.chdir(fileLoc)

df = df_iso_temp.sort_index(ascending = True)

df.to_csv(r'./snowCorePeakDepths.txt',header=True,index=True,sep = ' ')

# focusing on 2019

df_iso_temp = df_iso[(df_iso.peaks == 1)&(df_iso.year == 2019)].depthAcc_reg;
df = df_iso_temp.sort_index(ascending = True)
df.to_csv(r'./snowCorePeakDepths2019.txt',header=True,index=True,sep = ' ')

# insert these data
addListNames = ['SP1_20190611', 'SP1_20190626', 'SP1_20190715', 'SP1_20190724']
addListDepths = [18.1, 29.1, 31.3, 26.9]

addDict = dict(zip(addListNames,addListDepths))

# this loop adds the new peaks to the data set
for k in addDict.keys():

    p = int(k[2:3])
    y = int(k[4:8])
    m = int(k[8:10])
    d = int(k[10:12])
    peakAdd = df_iso[(df_iso.year == y) & (df_iso.month == m)&(df_iso.day == d)&
                     (df_iso.depthAcc_reg == addDict[k])&(df_iso.coreID == p)].peaks
    df_iso.loc[peakAdd.index,'peaks'] = 1

    print(k)
    print(df_iso.loc[peakAdd.index,'peaks'])
#    df_iso.loc[depth[maxMin].index,'peaks'] = 1            



# remove these data
subtractListNames = ['SP1_20190529', 'SP1_20190626']
subtractListDepths = [90.7, 6.0]

subtractDict = dict(zip(subtractListNames,subtractListDepths))

# this loop will take away peaks
for k in subtractDict.keys():

    p = int(k[2:3])
    y = int(k[4:8])
    m = int(k[8:10])
    d = int(k[10:12])
    peakAdd = df_iso[(df_iso.year == y) & (df_iso.month == m)&(df_iso.day == d)&
                     (df_iso.depthAcc_reg == subtractDict[k])&(df_iso.coreID == p)].peaks
    df_iso.loc[peakAdd.index,'peaks'] = 0

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
    
    
    for c in coreID[0:1]:  
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
            block[len(block)] = block[len(block)-1]+1;
            
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
            fm.plotProfile1(d,numDates,i,iso18O,brksTemp,hrsTemp,-1*depth,titleStr,'d18O','depth (cm)',[-50,-20],[-100,15])
            plt.plot(iso18O[maxMin],-depth[maxMin],'x',color = 'orange')
            i = i + 1;

        plt.show()

# fill in the time scale values for each profile

for y in yearUnique[-1:]:
    
    
    for c in coreID[0:1]:  
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
                    




# build a dataframe with the block stats 
pos = 1; #np.arange(1,6) for later when we've done the other positions
years = 2019; # np.arange(2017,2020) for later when we've done the other years
dates = df_iso[df_iso.year == years].date.unique()

cols = ['block','date','position','d18O','d18O_std','dD','dD_std','dexcess','dexcess_std','dxsln','dxsln_std']
df_blockStats = pd.DataFrame(columns = cols); # can start here with data frame only if doing one year

for d in dates:
    
#    for p in pos: don't need to iterate around position yet
    p = pos;
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
for d in dates:

    plt.figure()
    plt.xlim([-50,-20])
    plt.ylim([-100,15])
    plt.grid()
    alphas = np.arange(0.2,0.8,0.1) 

    for a,b in zip(alphas, rows):
        depth = -df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].depthAcc_reg
        d18O = df_iso[(df_iso.date == d)&(df_iso.coreID == p)&(df_iso.block == b)].d18O   
    #    fm.plotProfile3(d18O,depth,clr,d18Oblockvals)
        plt.plot(d18O,depth,'.',color = 'black')
        plt.axhspan(min(depth), max(depth), alpha=a, color='gray', zorder=0)
        blockVal = df_blockStats[(df_blockStats.block == b)&(df_blockStats.date == d)].d18O.values
        plt.text(-30,np.mean(depth),str(np.round(blockVal)))
    plt.title('eastGRIP pos = ' + str(p) + ', ' + str(d))
    plt.xlabel('d18O (per mil)')
    plt.ylabel('depth (cm)')


#
# temporary plotting script to plot the time values along the vertical axis


    plt.figure()
    plt.xlim([-50,-20])
#    plt.ylim([-100,15])
    plt.grid()
    alphas = np.arange(0.2,0.8,0.1) 

    for a,b in zip(alphas, rows):
        tsPlt = dfDay[(dfDay.coreID == p)&(dfDay.block == b)].timeScale
        d18O = dfDay[(dfDay.coreID == p)&(dfDay.block == b)].d18O   
    #    fm.plotProfile3(d18O,depth,clr,d18Oblockvals)
        plt.plot(d18O,tsPlt,'.',color = 'black')
        plt.axhspan(min(tsPlt), max(tsPlt), alpha=a, color='gray', zorder=0)
        blockVal = df_blockStats[(df_blockStats.block == b)&(df_blockStats.date == d)].d18O.values
        plt.text(-30,min(tsPlt)+0.5*(max(tsPlt)-min(tsPlt)),str(np.round(blockVal)))   
    plt.xlim([-50,-20])
    plt.grid()
    plt.title('eastGRIP pos = ' + str(p) + ', ' + str(d))
    plt.xlabel('d18O (per mil)')
    plt.ylabel('depth (cm)')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:42:05 2022

This code will cycle through all the curated EastGRIP data and put it on a regular depth
scale, then save the new data frame.

applies a shift to the 2018 data due to some confusion in the field as to which position was which.
see n_20221019_sfcTransectMet_profile2.pdf for details on why to subtract/add core depths in this 
adjustment proc.

@author: michaeltown
"""
# libraries
import pandas as pd
import EastGRIPprocs as eg
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime as dt
import figureMagic as fm


### main ###


## file locations and file names

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
vers = input('Which version of the df finalize (m = mean acc model, c = explicit acc model): ')

if vers.lower() == 'm':
    fileNameOG = 'eastGRIP_SCisoData_2016-2019_acc_peaks_t2.pkl'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t2.pkl'
    modelStr = 't2'

elif vers.lower() == 'c':
    fileNameOG = 'eastGRIP_SCisoData_2016-2019_acc_peaks_t3.pkl'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t3.pkl'
    modelStr = 't3'

# load df
df_iso = pd.read_pickle(fileLoc+fileNameOG)

# columns to keep
# cols = ['d18O','d18O_std','dD','dD_std','dexcess','dxsln']
cols = ['d18O','d18O_std','dD','dD_std','dexcess','dxsln','snowDatedt']
# create the indices
depthGrid = np.arange(-20,101,1)        # depth grid
coreIDs = np.arange(1,7,1)              # core IDs

dates = df_iso[(df_iso.year == 2017)|
               (df_iso.year == 2018)|
               (df_iso.year == 2019)].date.unique()            # dates of core extraction
dates.sort()
indexNames = ['depth','coreID','date']
multiInd = pd.MultiIndex.from_product([depthGrid,coreIDs,dates],names = indexNames)

# create the final DataFrame with multiindex 
df_iso_depth = pd.DataFrame(np.nan,index = multiInd,columns = cols)
dvals = df_iso_depth.index.get_level_values('depth').unique()

# this loop interpolates each profile to the dval grid and saves in df_iso_depth
for d in dates:
    
    for c in coreIDs[0:5]:
        
        dfTemp = eg.profPullEG_d(df_iso, d, c)
        dfTemp = dfTemp[cols]
        
        # get dates ready for interpolation
        
        if len(dfTemp) > 35:        # filters out short or dates where cores don't exist at certain positions
            dfTemp.snowDatedt = eg.convertDTtoDecimalYear(dfTemp.snowDatedt)
            
            dfTempInt = pd.DataFrame(columns = cols)
            dfTempInt = np.nan
    #        dfTempInt[cl] = np.interp(depthGrid,dfTemp.index,dfTemp[cl],left = np.nan,right = np.nan)
            # interp into each column at specified depth grid
            dfTempInt = eg.dfInterp(dfTemp,depthGrid,'depth')
            
            # convert interpolated decimal dates back to datetime format
            dfTemp2 = pd.DataFrame(dfTempInt.snowDatedt.dropna())
            dfTempInt.loc[dfTemp2.index,'snowDatedt'] = eg.convertDecimalYeartoDT(dfTemp2.snowDatedt)
            dfTempInt.snowDatedt = dfTempInt.snowDatedt.apply(lambda x: pd.to_datetime(x))
            
            # fill column multiindex  
            df_iso_depth.loc[(dvals,c,d),cols] = dfTempInt.values
'''            
            # checking the work
            fig = plt.figure()
            plt.plot(df_iso_depth.loc[(dvals,c,d),'d18O'],-depthGrid)
            plt.xlabel('d18O')
            plt.ylabel('depth (cm)')
            plt.title(str(d)[0:10]+' p: ' + str(c))
'''
        

# shift the profiles from 2018 into the proper locations and depths based on 
# conversations with Sonja Wahl and HCSL. See rn_20221019_sfcTransectMet_profile2.pdf

cols_shift= ['addMinSub']
coreIDs2 = np.arange(2,6)            # shifts start at position 2 and go through position 6
dates_shift = dates[22:25]
dSub = [[4.4,7.4,1.8,5.5],[4.9,7.4,2.3,6],[11.9,12.4,6.3,9]];   # this is the subtract from 2, 3, 4, 5 after shift
dAdd = [[7.4,3.3,7.5,0],[7.9,3.3,8,0.5],[14.9,8.3,12,3.5]];       # this is the add to 3, 4, 5 after shift and subtract
dictAddminSub = dict(zip(dates_shift,np.round(np.asarray(dAdd)-np.asarray(dSub),0)))


indexNames_shift = ['date','coreID']
multiInd_shift = pd.MultiIndex.from_product([dates_shift,coreIDs2],names = indexNames_shift)
df_shift = pd.DataFrame(index = multiInd_shift,columns = cols_shift)

for d in dates_shift:
    for i in np.arange(len(coreIDs2)):
        df_shift.loc[(d,coreIDs2[i]),'addMinSub'] = dictAddminSub[d][i]

df_iso_depth2 = df_iso_depth.copy()

# apply first the position shift, which is move the cores from one horizontal location to the next
for d in dates_shift:
    for c in np.arange(6,2,-1):
        for cl in cols:
            df_iso_depth2.loc[(dvals,c,d),cl] = df_iso_depth.loc[(dvals,c-1,d),cl].values
            df_iso_depth2.loc[(dvals,c-1,d),cl] = np.nan

# apply the second shift, which is adjust the height of the cores in their new position

df_iso_depth3 = pd.DataFrame(np.nan,index = multiInd,columns = cols)

for d in dates_shift:
    for c in np.arange(6,2,-1):
        for cl in cols:
            # subtract the 'add minus subtract' values because the dvals are positive downwards
            dvalsNew = dvals-df_shift.loc[(d,c-1),'addMinSub']
            dvalsDict = dict(zip(dvals,dvalsNew))
            for dv in dvalsDict.keys():
                if ~pd.isnull(df_iso_depth.loc[(dv,c,d),cl]) :
                    df_iso_depth3.loc[(dvalsDict[dv],c,d),cl] = df_iso_depth2.loc[(dv,c,d),cl]

# assign the rest of df_iso_depth2 to df_iso_depth3

# first find a list of the dates not in this problem
dates3 = np.copy(dates)

for d in dates_shift:
    if d in dates3:
        dates3 = np.delete(dates3,np.where(dates3 == d))
        
for d in dates3:
    for c in coreIDs:
        for cl in cols:
            df_iso_depth3.loc[(dvals,c,d),cl] = df_iso_depth2.loc[(dvals,c,d),cl].values

for d in dates_shift:
        for cl in cols:
            df_iso_depth3.loc[(dvals,1,d),cl] = df_iso_depth.loc[(dvals,1,d),cl].values
    

    

# plot all data pairs such that each date has three values on it, and then double 
# check the shifted positions and shifted vertical values

for c in np.arange(3,7,1):
    for d in dates_shift:
        fig = plt.figure()
        plt.plot(df_iso_depth.loc[(dvals,c-1,d),'d18O'],-dvals,color = 'black',
                 alpha = 0.5, linewidth = 9)
        plt.plot(df_iso_depth2.loc[(dvals,c,d),'d18O'],-dvals,color = 'black')
        plt.plot(df_iso_depth3.loc[(dvals,c,d),'d18O'],-dvals,color = 'red',
                 alpha = 0.5)
        plt.title(str(d)[0:10] + ', p = ' + str(c))
        plt.ylim([-100,20])
        plt.xlim([-50,-20])
        
# check all plots together for consistency
dates2 = df_iso[(df_iso.year == 2018)].date.unique()            # dates of core extraction
dates2.sort()

for c in np.arange(1,7,1):
    for d in dates2:
        fig = plt.figure()

        if ((c < 3)|(d < dates2[4])):
            plt.plot(df_iso_depth.loc[(dvals,c,d),'d18O'],-dvals,color = 'black',alpha = 0.5, 
                     linewidth = 9, label = 'no move p, no adj, p = ' + str(c-1))            
            plt.plot(df_iso_depth2.loc[(dvals,c,d),'d18O'],-dvals,color = 'black', 
                     label = 'move p, no adj, p = ' + str(c), linestyle = '--')
            plt.plot(df_iso_depth3.loc[(dvals,c,d),'d18O'],-dvals,color = 'red',
                     label = 'move p, acc adj, p = ' + str(c),alpha = 0.5)
        elif ((c >= 3)&(d >= dates2[4])):
            plt.plot(df_iso_depth.loc[(dvals,c,d),'d18O'],-dvals,color = 'black',
                     alpha = 0.5, linewidth = 9, 
                     label = 'no move p, no adj, p = ' + str(c-1))
            plt.plot(df_iso_depth2.loc[(dvals,c,d),'d18O'],-dvals,color = 'black', 
                     label = 'move p, no adj, p = ' + str(c),linestyle = '--')
            plt.plot(df_iso_depth3.loc[(dvals,c,d),'d18O'],-dvals,color = 'red', 
                     label = 'move p, acc adj, p = ' + str(c),alpha = 0.5)
        plt.title(str(d)[0:10])
        plt.ylim([-100,20])
        plt.xlim([-50,-20])
        plt.legend(loc= 'lower left')
        
    

# final plot series of df_iso_depth3
dates = pd.DatetimeIndex(dates)
for c in np.arange(1,7,1):
    for d in dates:
        fig = plt.figure()
        plt.plot(df_iso_depth3.loc[(dvals,c,d),'d18O'],-dvals,alpha = 0.5)
        plt.title(str(d)[0:10]+ ', p = ' + str(c) + ', ' + modelStr + ' model')
        plt.ylim([-100,20])
        plt.xlim([-50,-20])
        plt.xlabel('d18O (per mil)')
        plt.ylabel('depth (cm)')
        if d.year == 2017:
            fm.saveas(fig,figureLoc+'2017/d18OvsDepth_p'+str(c)+'_'+str(d)[0:10]+'_'+modelStr+'_2017')
        elif d.year == 2018:
            fm.saveas(fig,figureLoc+'2018/d18OvsDepth_p'+str(c)+'_'+str(d)[0:10]+'_'+modelStr+'_2018')
        elif d.year == 2019:
            fm.saveas(fig,figureLoc+'2019/d18OvsDepth_p'+str(c)+'_'+str(d)[0:10]+'_'+modelStr+'_2019')


# clean the index values of the extraneous indicies
dvalsExt = df_iso_depth3.index.get_level_values(0).unique()[len(dvals):]

for i in dvalsExt:
    df_iso_depth3.drop(i,level = 0,axis = 0, inplace = True)

# pickle the final data file
df_iso_depth3.to_pickle(fileLoc+fileNameNew)
df_iso_depth_flat = df_iso_depth3.reset_index()
df_iso_depth_flat.to_csv(fileLoc+fileNameNew[:-4]+'.csv')



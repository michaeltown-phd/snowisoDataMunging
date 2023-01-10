#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:04:33 2022

This code will take the depth indexed data and interpolate to a regular time grid
of 3-day resolution.

@author: michaeltown
"""

# libraries
import numpy as np
import pandas as pd
import datetime as dt
import os as os
import EastGRIPprocs as eg
import matplotlib.pyplot as plt
import figureMagic as fm

# file load
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
vers = input('Which version of the df finalize (m = mean acc model, c = explicit acc model): ')
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'


if vers.lower() == 'm':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t2.pkl'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_tindex_accExt_t2.pkl'
    modelStr = 't2'


elif vers.lower() == 'c':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_dindex_accExt_t3.pkl'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_tindex_accExt_t3.pkl'
    modelStr = 't3'

df_iso = pd.read_pickle(fileLoc+fileNameOG)

# make the time series
beg = pd.to_datetime('20130101',format = '%Y%m%d')
end = pd.to_datetime('20200101',format = '%Y%m%d')

ageDepth = pd.date_range(beg,end,freq = '1D')         # choose a freq of 1D so cores can always
                                                # start on the right date
                                                
# prep df for interpolation
# these are the values from the depth-based index
depthVals = df_iso.index.get_level_values('depth').unique()
coreIDs = df_iso.index.get_level_values('coreID').unique()
coreDateVals = df_iso.index.get_level_values('date').unique()

# make the new data frame
indexNames = ['ageDepth','coreID','date']
colsOld = ['d18O','d18O_std','dD','dD_std','dexcess','dxsln','snowDatedt']
colsNew = ['depth','d18O','d18O_std','dD','dD_std','dexcess','dxsln']
multiInd = pd.MultiIndex.from_product([ageDepth,coreIDs,coreDateVals],names = indexNames)
df_iso_time = pd.DataFrame(np.nan,index = multiInd,columns = colsNew)


# set the columns of the new data frame
interpMethod = 'linear'

for c in coreIDs:
    for date in coreDateVals:

        dfTemp = eg.profPullEG_t_mi(df_iso,depthVals,date,c,colsOld)

        if len(dfTemp) > 35:        # filters out short or dates where cores don't exist at certain positions

            # interpolate to 1-day grid
            dfTempInt = eg.dfInterpTime(dfTemp,'1D','linear')
            dfTempInt = dfTempInt.drop(columns = ['coreID'])
            # limit of data to assign
            beg = np.min(dfTempInt.index)
            end = np.max(dfTempInt.index)
            ind = np.where((ageDepth >= beg)&(ageDepth <= end))
            # assign values to the 
            df_iso_time.loc[(ageDepth[ind],c,date),colsNew] = dfTempInt.values
            fig = plt.figure()
            plt.plot(df_iso_time.loc[(ageDepth,c,date),'d18O'],ageDepth)
            plt.xlabel('d18O')
            plt.ylabel('age depth')
            plt.xlim([-50,-20])
            if date.year == 2019:
                beg = pd.to_datetime('20160901',format = '%Y%m%d')
                end = pd.to_datetime('20190901',format = '%Y%m%d')
                plt.ylim([beg,end])
            elif date.year == 2018:
                beg = pd.to_datetime('20150901',format = '%Y%m%d')
                end = pd.to_datetime('20180901',format = '%Y%m%d')
                plt.ylim([beg,end])
            elif date.year == 2017:
                beg = pd.to_datetime('20140901',format = '%Y%m%d')
                end = pd.to_datetime('20170901',format = '%Y%m%d')
                plt.ylim([beg,end])

            plt.title(str(date)[0:10]+' p: ' + str(c))
            if date.year == 2017:
                fm.saveas(fig,figureLoc+'2017/d18OvsTime_'+str(date)[0:10]+'_'+modelStr+'_2017')
            elif date.year == 2018:
                fm.saveas(fig,figureLoc+'2018/d18OvsTime_'+str(date)[0:10]+'_'+modelStr+'_2018')
            elif date.year == 2019:
                fm.saveas(fig,figureLoc+'2019/d18OvsTime_'+str(date)[0:10]+'_'+modelStr+'_2019')


df_iso_time.to_pickle(fileLoc+fileNameNew)


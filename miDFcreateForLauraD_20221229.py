#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:32:44 2022

This code should create an MI-DF of the csv files of EastGRIP observations
from 2017-2019

@author: michaeltown
"""

# libraries
import pandas as pd
import EastGRIPprocs as eg
import numpy as np
import matplotlib.pyplot as plt

# file locations

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/pangaeaSubmission/'

vers = input('Which version of the df finalize (m = mean acc model, c = explicit acc model): ')

if vers.lower() == 'm':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_depthIndex_meanAccModelBot.csv'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_depthIndex_meanAccModelBot.pkl'
    modelStr = vers

elif vers.lower() == 'c':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_depthIndex_manualConstModelBot.csv'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_depthIndex_manualConstModelBot.pkl'
    modelStr = vers


# load and clean df
df_iso = pd.read_csv(fileLoc+fileNameOG)
df_iso['date']= pd.to_datetime(df_iso.dateOfExtraction)
df_iso['year'] = df_iso.date.dt.year

df_iso.drop('Unnamed: 0', axis = 1,inplace = True)
df_iso.drop('dateOfExtraction', axis = 1,inplace = True)

colsOld = ['depth (cm)', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)',
       'dD_std (per mille)', 'dexcess (per mille)', 'dxsln (per mille)',
       'ageDepth']
colsNew = ['depthAcc_reg','d18O','d18O_std','dD','dD_std','dexcess','dxsln','snowDatedt']
df_iso.rename(columns = dict(zip(colsOld,colsNew)),inplace = True)


# columns to keep
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
    
    for c in coreIDs:
        
        dfTemp = eg.profPullEG_d(df_iso, d, c)
        dfTemp = dfTemp[cols]
        df_iso_depth.loc[(dvals,c,d),cols] = dfTemp.values
        # checking the work
        fig = plt.figure()
        plt.plot(df_iso_depth.loc[(dvals,c,d),'d18O'],-depthGrid)
        plt.xlabel('d18O')
        plt.ylabel('depth (cm)')
        plt.title(str(d)[0:10]+' p: ' + str(c))


#####
# time-based df
#####


# file load
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/pangaeaSubmission/'
vers = input('Which version of the df finalize (m = mean acc model, c = explicit acc model): ')


if vers.lower() == 'm':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_timeIndex_meanAccModelBot.csv'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_timeIndex_meanAccModelBot.pkl'
    modelStr = vers

elif vers.lower() == 'c':
    fileNameOG = 'eastGRIP_SCisoData_2017-2019_timeIndex_manualConstModelBot.csv'
    fileNameNew = 'eastGRIP_SCisoData_2017-2019_timeIndex_manualConstModelBot.pkl'
    modelStr = vers

# load and clean df
df_iso = pd.read_csv(fileLoc+fileNameOG)
df_iso['date']= pd.to_datetime(df_iso.dateOfExtraction)
df_iso['year'] = df_iso.date.dt.year

df_iso.drop('Unnamed: 0', axis = 1,inplace = True)
df_iso.drop('dateOfExtraction', axis = 1,inplace = True)

colsOld = ['depth (cm)', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)',
       'dD_std (per mille)', 'dexcess (per mille)', 'dxsln (per mille)',
       'ageDepth']
colsNew = ['depth','d18O','d18O_std','dD','dD_std','dexcess','dxsln','snowDatedt']
df_iso.rename(columns = dict(zip(colsOld,colsNew)),inplace = True)



# make the time series
beg = pd.to_datetime('20130101',format = '%Y%m%d')
end = pd.to_datetime('20200101',format = '%Y%m%d')

ageDepth = pd.date_range(beg,end,freq = '1D')         # choose a freq of 1D so cores can always
                                                # start on the right date
                                                
# make the new data frame
indexNames = ['ageDepth','coreID','date']
colsOld = ['d18O','d18O_std','dD','dD_std','dexcess','dxsln','snowDatedt']
colsNew = ['depth','d18O','d18O_std','dD','dD_std','dexcess','dxsln']
multiInd = pd.MultiIndex.from_product([ageDepth,coreIDs,coreDateVals],names = indexNames)
df_iso_time = pd.DataFrame(np.nan,index = multiInd,columns = colsNew)

# prep df for interpolation
# these are the values from the depth-based index
depthVals = df_iso.depth.unique()
coreIDs = df_iso.coreID.unique()
coreDateVals = pd.to_datetime(df_iso.date.unique())


for c in coreIDs:
    for date in coreDateVals:

        dfTemp = eg.profPullEG_t(df_iso,date,c)

        # limit of data to assign
        beg = np.min(dfTemp.snowDatedt)
        end = np.max(dfTemp.snowDatedt)
        ind = np.where((ageDepth >= beg)&(ageDepth <= end))

        df_iso_time.loc[(ageDepth[ind],c,date),colsNew] = dfTemp[colsNew].values

        # plot the data to make sure it came out ok
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

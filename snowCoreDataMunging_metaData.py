#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:57:34 2022

@author: michaeltown
"""

'''
data munging code to read in the snow core meta data from EastGRIP 
this code will 
1. read in the meta data from each field season
2. normalize all the file headers and 

NOTES: there are no meta data of this sort for 2016

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from pandas_ods_reader import read_ods
import pickle as pkl


# locations of each file

metaFile2017 = '/home/michaeltown/work/projects/snowiso/EastGRIP2017/'
metaFile2018 = '/home/michaeltown/work/projects/snowiso/EastGRIP2018/2018FieldWork/'
metaFile2019 = '/home/michaeltown/work/projects/snowiso/EastGRIP2019/'


# 2019 data

fileName = 'EastGRIP_snowpack_overview_2019.ods'
dfDict = pd.read_excel(metaFile2019+fileName,sheet_name = None)

sheetNames = list(dfDict.keys());
df_EGRIP_SCmeta = dfDict[sheetNames[0]];

# create list of snow core names
dates = df_EGRIP_SCmeta.Date.unique()
snowCoreNames = []

for d in dates:
    for i in np.arange(1,6):
        snowCoreNames.append('SP'+str(i)+'_'+d.replace('.',''))

# fill in the data frame with 2019 data

df_EGRIP_SCmeta['Breaks'] = df_EGRIP_SCmeta['Breaks'].str.split(',');
df_EGRIP_SCmeta['Hoar Layer'] = df_EGRIP_SCmeta['Hoar Layer'].str.split(',');
df_EGRIP_SCmeta['snowCoreName'] = snowCoreNames;

# rename columns
oldNames = ['Snow stick height (cm)   ','Hoar Layer','Breaks','Accumulation'];
newNames = ['stickHeight','hoar','breaks','accumulation'];
df_EGRIP_SCmeta.rename(columns = dict(zip(oldNames,newNames)),inplace=True);
 
# drop unneeded columns
df_EGRIP_SCmeta.drop(['Snowpack','SP Length','Missing snow at bottom (cm)','# sample bags','Date','Time','dAccumulation'], axis = 1, inplace = True);


####################################
# 2018 data
####################################
fileName = 'EastGRIP_snowpack_overview_2018.ods'
dfDict = pd.read_excel(metaFile2018+fileName,sheet_name = None)

sheetNames = list(dfDict.keys());
df_EGRIP_SCmetaTemp = dfDict[sheetNames[0]];

for s in sheetNames[1:]:
    df_EGRIP_SCmetaTemp = df_EGRIP_SCmetaTemp.append(dfDict[s]);

df_EGRIP_SCmetaTemp['breaks'] = df_EGRIP_SCmetaTemp['Breaks'].str.split(',');
#df_EGRIP_SCmetaTemp.drop(['Breaks'], axis = 1, inplace = True)
df_EGRIP_SCmetaTemp['hoar'] = df_EGRIP_SCmetaTemp['Hoar Layer'].str.split(',');
df_EGRIP_SCmetaTemp.drop(['Hoar Layer'], axis = 1, inplace = True)

# rename columns
oldNames = ['Sample name', 'Measured height', 'Accumulation','Notes','Responsible'];
newNames = ['snowCoreName','stickHeight','accumulation','notes','who']
df_EGRIP_SCmetaTemp.rename(columns = dict(zip(oldNames,newNames)),inplace=True);

# drop columns that are unnecessary
columnsDrop = ['Core length', 'Sample #', 'Box Num','Date','dAccumulation'];
df_EGRIP_SCmetaTemp.drop(columnsDrop, axis = 1, inplace = True)


####################################
# 2017 data
####################################

fileName = 'EastGRIP_overview_snowpack_2017.ods'
dfDict = pd.read_excel(metaFile2017+fileName,sheet_name = None)

sheetNames = list(dfDict.keys());
df_EGRIP_SCmetaTemp2 = dfDict[sheetNames[0]];

for s in sheetNames[1:]:
    df_EGRIP_SCmetaTemp2 = df_EGRIP_SCmetaTemp2.append(dfDict[s]);

df_EGRIP_SCmetaTemp2['breaks'] = df_EGRIP_SCmetaTemp2['Breaks'].str.split(',');
df_EGRIP_SCmetaTemp2.drop(['Breaks'], axis = 1, inplace = True)
df_EGRIP_SCmetaTemp2['hoar'] = df_EGRIP_SCmetaTemp2['Hoar Layer'].str.split(',');
df_EGRIP_SCmetaTemp2.drop(['Hoar Layer'], axis = 1, inplace = True)

# rename columns
oldNames = ['Sample name', 'Measured height', 'Accumulation','Notes','Responsible'];
newNames = ['snowCoreName','stickHeight','accumulation','notes','who']
df_EGRIP_SCmetaTemp2.rename(columns = dict(zip(oldNames,newNames)),inplace=True);

# drop columns that are unnecessary
columnsDrop = ['Core length', 'Sample #','Date','dAccumulation'];
df_EGRIP_SCmetaTemp2.drop(columnsDrop, axis = 1, inplace = True)

# join all dfs
df_EGRIP_SCmeta = df_EGRIP_SCmeta.append(df_EGRIP_SCmetaTemp);
df_EGRIP_SCmeta = df_EGRIP_SCmeta.append(df_EGRIP_SCmetaTemp2);
df_EGRIP_SCmeta = df_EGRIP_SCmeta.set_index('snowCoreName')


# save the meta data
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/')
dataFileName = 'eastGRIP_metaData_2017-2019.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_SCmeta,outfile);
outfile.close();


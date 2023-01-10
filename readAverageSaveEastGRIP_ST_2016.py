#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:00:37 2022

read and make daily averages of the surface transect data from 2016

@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import os as os

### main ###

# load the data files
fileLoc = '/home/michaeltown/work/projects/snowiso/EastGRIP2016/surfaceTransect/'
dirList = os.listdir(fileLoc)

dirList = [d for d in dirList if '.txt' in d]
dirList = [d for d in dirList if 'flag' not in d]

colsTemp = ['sampleName','d18O','dD','dexcess']
cols = ['d18O','d18O_std','dD','dD_std','dexcess']

dates = [pd.to_datetime('20'+d[2:8],format = '%Y%m%d') for d in dirList]
dates.sort()
df_st = pd.DataFrame(index = dates, columns = cols)

for d in dirList:
    
    dfTemp = pd.read_csv(fileLoc+d,sep = '\t')
    dfTemp.columns = colsTemp
    dfTemp.set_index('sampleName',inplace = True)
    
    date = pd.to_datetime('20'+d[2:8],format = '%Y%m%d')
    df_st.loc[date,'d18O'] = np.mean(dfTemp.d18O)
    df_st.loc[date,'d18O_std'] = np.std(dfTemp.d18O)
    df_st.loc[date,'dD'] = np.mean(dfTemp.dD)
    df_st.loc[date,'dD_std'] = np.std(dfTemp.dD)
    df_st.loc[date,'dexcess'] = np.mean(dfTemp.dexcess)

df_st.to_pickle(fileLoc+'surfaceTransectDataEastGRIP2016.pkl')
df_st.to_csv(fileLoc+'surfaceTransectDataEastGRIP2016.csv')

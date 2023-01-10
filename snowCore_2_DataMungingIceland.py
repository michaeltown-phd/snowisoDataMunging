#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:31:49 2022

@author: michaeltown
"""
'''
reads and processes EastGRIP 2016-2019 data from the iceland lab
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from pandas_ods_reader import read_ods
import pickle as pkl

def sampleNameRefine(s):
    
    newS = s;
    
    if '_' in s[-2:]:
        newS = s[:-2]+'_0'+s[-1]
    
    return newS
        

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'

fileNames = ['SP_2017_Iceland.ods','SP_2018_Iceland.ods','SP_2019_Iceland.ods']
#fileNames = ['SP_2018_Iceland.ods']

cols = ['Sample_nr', 'd18', 'd18_std', 'dD', 'dD_std']
df_EGRIP_iceland = pd.DataFrame(columns=cols)


for f in fileNames:

    # create dict for the specific year of data from iceland data set
    # dfDict = pd.read_excel(fileLoc+fileName, sheet_name=None)
    dfDict = pd.read_excel(fileLoc+f, sheet_name=None)
    
    sheetNames = list(dfDict.keys())
    
    # loop through all the sheets
    
    for s in sheetNames:
    
        df_EGRIP_SC = dfDict[s]
        df_EGRIP_SC.dropna(axis=1, how='all', inplace=True)
        
        # drop the comments column (but be sure to check later)
        colTemp = df_EGRIP_SC.columns;
        colTemp = [c for c in colTemp if 'Comment' in c];
        df_EGRIP_SC = df_EGRIP_SC.drop(columns = colTemp)
    
     
    
        # concatenate columns with similar names, need to strip the tailing .# off of each name
    
        i = 0
        colsLoop = list(df_EGRIP_SC.columns)
        
        # need to treat condition where there are not samples from site 1 initially
        # so rename the first 5 of these columns by truncating the .# off 
        if len(colsLoop[0]) == 11:
            colTempRenameDict = dict(zip(colsLoop[0:5], [x[:-2] for x in colsLoop[0:5]]))
            df_EGRIP_SC = df_EGRIP_SC.rename(columns=colTempRenameDict)
            colsLoop[0:5] = [x[:-2] for x in colsLoop[0:5]]
            
        
        
        # while loop to go through each set of 5 columns in each sheet
        while i < len(colsLoop):
    
            if i+5 > len(colsLoop):
                c = colsLoop[i:]
            else:
                c = colsLoop[i:i+5]
    
            # strip the final ends of each set of five columns to append to the master data frame
    
            if i > 0:
                tempDictCols = dict(zip(c, [x[:-2] for x in c]))
    
                if i+5 > len(colsLoop):
                    dfTemp = df_EGRIP_SC[colsLoop[i:]]
                else:
                    dfTemp = df_EGRIP_SC[colsLoop[i:i+5]]
    
                dfTemp = dfTemp.rename(columns=tempDictCols)
            else:
                dfTemp = df_EGRIP_SC[colsLoop[i:i+5]]
    
            df_EGRIP_iceland = df_EGRIP_iceland.append(dfTemp)
    
            i = i + 5



# convert the entries to numbers by replacing , with . and then turning to number
cols = ['d18','d18_std','dD','dD_std']
for c in cols:
    df_EGRIP_iceland[c] = df_EGRIP_iceland[c].str.replace(',','.').astype('float')


df_EGRIP_iceland = df_EGRIP_iceland.dropna(subset=['Sample_nr'])
print(df_EGRIP_iceland.info())

cols = ['Sample_nr','d18','d18_std']
colsNew = ['Sample','d18O','d18O_std']

df_EGRIP_iceland = df_EGRIP_iceland.rename(columns = dict(zip(cols,colsNew)))

# fix the indexing of the sample names to be uniform

df_EGRIP_iceland['Sample'] = df_EGRIP_iceland['Sample'].apply(sampleNameRefine)


# save the data
os.chdir(fileLoc)
dataFileName = 'eastGRIP_SCisoData_2017-2019_iceland.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_iceland,outfile);
outfile.close();



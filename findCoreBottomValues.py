#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:33:54 2022

This code will read in the primary snowcore data frame and find the bottoms of each
core: d18O content, depth, and sample number. 

The purpose of this task is to explicitly assign the bottom depth of each core a date
in the depth-age model. This step is a debateable choice in constraining the depth-age
model of each core, rather than letting mean accumulation rate dictate the depth-age
model for the bottom of each core.

@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np

# load data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileName = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'

df = pd.read_pickle(fileLoc + fileName)

# loop through each core can find the bottom value
valList = [];
years = np.arange(2017,2020)
coreIDs = np.arange(1,6)


for y in years:
    for c in coreIDs:
        dates = df[(df.year == y)&(df.coreID == c)].date.unique()
        for d in dates:
            maxD = np.max(df[(df.year == y)&(df.coreID == c)&(df.date == d)].depthAcc)
            d18O = df[(df.year == y)&(df.coreID == c)&(df.date == d)&(df.depthAcc == maxD)].d18O
            sampleID = df[(df.year == y)&(df.coreID == c)&(df.date == d)&(df.depthAcc == maxD)].index
                                                                       
            tempList = [sampleID.values[0], d18O.values[0],maxD]
            valList.append(tempList)

file=open(fileLoc+'coreBottomValues2017-2019.txt','w')
for items in valList:
    file.writelines(items[0]+','+str(items[1])+','+str(np.round(items[2],2))+','+'\n')
file.close()



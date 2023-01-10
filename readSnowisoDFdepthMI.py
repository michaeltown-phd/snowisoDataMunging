#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:02:39 2022

This proc gives some examples on how to access the multi-index data frame
for the snowiso data (2017-2019) that is interpolated to a regular depth grid

@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileName = 'eastGRIP_SCisoData_2017-2019_dindex_accExt.pkl'

# load df
df_iso1 = pd.read_pickle(fileLoc+fileName)


# isolate the index values for easier access (math and plotting)
dvals = df_iso1.index.get_level_values('depth').unique()
dates = df_iso1.index.get_level_values('date').unique()
coreIDs = np.arange(1,6,1)
depthGrid = np.arange(20,-101,-1)        # depth grid


# example of how to access the multi-index profiles by plot the d18O profiles
            
for d in dates:
    for c in coreIDs:
        
            # this will not filter out short or non-existing core dates
        
            fig = plt.figure()
            plt.plot(df_iso1.loc[(dvals,c,d),'d18O'],depthGrid)
            plt.xlabel('d18O')
            plt.ylabel('depth (cm)')
            plt.title(str(d)[0:10]+' p: ' + str(c))

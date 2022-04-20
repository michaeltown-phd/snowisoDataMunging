#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:08:20 2022

A script to plot mean daily cores for intercomparison of features as a function of time through a season.

@author: michaeltown
"""

#libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import pickle as pkl
from scipy.signal import find_peaks
import figureMagic as fm


# useful stuff
#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'ppt' # '\textperthousand'


# read data from files


fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'

df_iso = pd.read_pickle(fileLoc+fileNameIso);


# plot average of each day (1-5 positions) as separate subplots, with std, num
# this groupby gives a multiindex data frame.


columnsToProcess = ['d18O','dD','dexcess','dxsln']
yearUnique = df_iso.year.unique();

df_iso_pos = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].mean()
df_iso_pos[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].std()
df_iso_pos[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].max()
df_iso_pos[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].min()
df_iso_pos[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].count()

df_iso_pos = df_iso_pos.reset_index(level = 0);         # should reset the index to date and leave depthAcc_reg as a column
df_iso_pos['year'] = df_iso_pos.index.year
df_iso_pos['dates'] = df_iso_pos.index.date

fileLoc = None;


for y in yearUnique[1:]:

    dates = df_iso_pos[df_iso_pos.year == y].dates.unique()
    dates.sort()
    
    for d in dates:
        
        figureLocTemp = figureLoc + str(y) +'/'
        
        num = df_iso_pos[df_iso_pos.dates == d].d18O_num
        iso = df_iso_pos[df_iso_pos.dates == d].d18O
        lbstd = df_iso_pos[df_iso_pos.dates == d].d18O-df_iso_pos[df_iso_pos.dates == d].d18O_std;
        ubstd = df_iso_pos[df_iso_pos.dates == d].d18O+df_iso_pos[df_iso_pos.dates == d].d18O_std;
        lbmin = df_iso_pos[df_iso_pos.dates == d].d18O_min;
        ubmax = df_iso_pos[df_iso_pos.dates == d].d18O_max;
        depth = df_iso_pos[df_iso_pos.dates == d].depthAcc_reg;
        fm.myDepthFunc(iso,-depth,num,'black',lbstd,ubstd,lbmin,ubmax,'EastGRIP ' + str(d) + ' '+d18Osym+' profile',
                    'd18O','depth (cm)',[-50,-20],[-100,15],figureLocTemp,'prof_d18O_dailyMean_'+str(d));

        num = df_iso_pos[df_iso_pos.dates == d].dD_num
        iso = df_iso_pos[df_iso_pos.dates == d].dD
        lbstd = df_iso_pos[df_iso_pos.dates == d].dD-df_iso_pos[df_iso_pos.dates == d].dD_std;
        ubstd = df_iso_pos[df_iso_pos.dates == d].dD+df_iso_pos[df_iso_pos.dates == d].dD_std;
        lbmin = df_iso_pos[df_iso_pos.dates == d].dD_min;
        ubmax = df_iso_pos[df_iso_pos.dates == d].dD_max;
        depth = df_iso_pos[df_iso_pos.dates == d].depthAcc_reg;
        fm.myDepthFunc(iso,-depth,num,'blue',lbstd,ubstd,lbmin,ubmax,'EastGRIP ' + str(d) + ' '+dDsym+' profile',
                    'dD','depth (cm)',[-380,-150],[-100,15],figureLocTemp,'prof_dD_dailyMean_'+str(d));

        num = df_iso_pos[df_iso_pos.dates == d].dexcess_num
        iso = df_iso_pos[df_iso_pos.dates == d].dexcess
        lbstd = df_iso_pos[df_iso_pos.dates == d].dexcess-df_iso_pos[df_iso_pos.dates == d].dexcess_std;
        ubstd = df_iso_pos[df_iso_pos.dates == d].dexcess+df_iso_pos[df_iso_pos.dates == d].dexcess_std;
        lbmin = df_iso_pos[df_iso_pos.dates == d].dexcess_min;
        ubmax = df_iso_pos[df_iso_pos.dates == d].dexcess_max;
        depth = df_iso_pos[df_iso_pos.dates == d].depthAcc_reg;
        fm.myDepthFunc(iso,-depth,num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EastGRIP ' + str(d) + ' dxs profile',
                    'dexcess','depth (cm)',[-10,30],[-100,15],figureLocTemp,'prof_dxs_dailyMean_'+str(d));

        num = df_iso_pos[df_iso_pos.dates == d].dxsln_num
        iso = df_iso_pos[df_iso_pos.dates == d].dxsln
        lbstd = df_iso_pos[df_iso_pos.dates == d].dxsln-df_iso_pos[df_iso_pos.dates == d].dxsln_std;
        ubstd = df_iso_pos[df_iso_pos.dates == d].dxsln+df_iso_pos[df_iso_pos.dates == d].dxsln_std;
        lbmin = df_iso_pos[df_iso_pos.dates == d].dxsln_min;
        ubmax = df_iso_pos[df_iso_pos.dates == d].dxsln_max;
        depth = df_iso_pos[df_iso_pos.dates == d].depthAcc_reg;
        fm.myDepthFunc(iso,-depth,num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EastGRIP ' + str(d) + ' dxsln profile',
                    'dxsln','depth (cm)',[-5,35],[-100,15],figureLocTemp,'prof_dxsln_dailyMean_'+str(d));

# plot the difference of from the mean as an evolution in time, contour plot of changes...
# need to develop a df of the mean iso values with column names as the date, and the index as the depth.
#        iso18Omean =
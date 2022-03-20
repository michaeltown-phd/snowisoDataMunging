#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:57:13 2022

@author: michaeltown
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from pandas_ods_reader import read_ods
import pickle as pkl


def plotScatter(iso1,iso2,clr,title,xlab,ylab,xlim,ylim,f):

    figO18dxs = plt.figure()
    plt.plot(iso1,iso2,'.',color = clr,alpha = 0.3);
    plt.grid()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(xlim,[0,0],'k--')
    figO18dxs.savefig('./'+f+'.jpg')  



# useful stuff
#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'ppt' # '\textperthousand'

# main

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);


#plot all d18O vs. all dexcess
os.chdir(figureLoc)
titleStr = 'd18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_')
figO18dxs = plt.figure()        
plotScatter(df_iso.d18O,df_iso.dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)


# plot year 0
dupper = 15;
dlower = -25;
titleStr = 'Year 0 (-25 to 15 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
figO18dxs = plt.figure()
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)


# plot summer
dupper = 15;
dlower = -5;
titleStr = 'Summer 0 (-5 to 15 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
figO18dxs = plt.figure()
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)

# plot spring
dupper = -5;
dlower = -10;
titleStr = 'Spring 0 (-10 to -5 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
figO18dxs = plt.figure()
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)

# plot winter
dupper = -10;
dlower = -20;
titleStr = 'Winter 0 (-20 to -10 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
figO18dxs = plt.figure()
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)

# plot Fall
dupper = -20;
dlower = -25;
titleStr = 'Fall 0 (-25 to -20 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
figO18dxs = plt.figure()
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)

# plot Summer
dupper = -25;
dlower = -35;
titleStr = 'Summer 1 (-35 to -25 cm) d18O vs dxs EastGRIP, 2016-2019'
fileNameFigure = titleStr.replace(' ','').replace(',','').replace('-','_').replace('(','_').replace(')','_')
d18O = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].d18O        
dexcess = df_iso[(-df_iso.depth> dlower)&((-df_iso.depth< dupper))].dexcess
plotScatter(d18O,dexcess,'blue',titleStr,d18Osym+' (ppt)', 'dxs (ppt)',[-50,-20],[-10,20],fileNameFigure)


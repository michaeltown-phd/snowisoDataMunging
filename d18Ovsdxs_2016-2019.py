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
import pickle as pkl
import simpleYork as york
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

def regressPipeline_dDd18O(x, y, ts,titleStr, xlab, ylab, xlim, ylim):
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = ts);


    model = sm.OLS(yTrain,xTrain);
    results = model.fit();
    
    
    yNew = results.predict(xTest);
    residTest = yNew - yTest;
    
    slopes = results.params[1];
    intercept = results.params.const;
    r2score= results.rsquared;
    
    # plot a figure

    fig1 = plt.figure();
    plt.plot(x.iloc[:,1],y,'.',color = 'blue',alpha = 0.2);
    plt.plot(xlim,slopes*xlim+intercept,'--',color = 'red',alpha = 0.5)
#    plt.plot(xlim,8*xlim,'--',color = 'black',alpha = 0.3)        # for equilibrium slope
    plt.title(titleStr);
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel(ylab);
    plt.xlabel(xlab);
    plt.grid();
    
    
    xval = xlim[0];
    if xval < 0:
        xval = xlim[1]*1.3
    else:
        xval = xlim[1]*0.8
    
    plt.text(xval, ylim[1]*0.8, 'm = ' + str(np.round(slopes,2)))
    plt.text(xval, ylim[1]*0.7, 'b = ' + str(np.round(intercept,2)))
    plt.text(xval, ylim[1]*0.6, 'r\u00b2 = ' + str(np.round(r2score,2)))

    # return the params
    return slopes, intercept, r2score


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

def plotProfileDaily(date,spid,dD,d18O,dxs,bks,hrs,depth,title):

    #set the breaks/hoar values to something that will appear on the graphs properly
    
    bks = [b*np.mean(iso) if b == 1 else np.nan for b in bks]
    hrs = [h*np.mean(iso) if h == 1 else np.nan for h in hrs]

    
    
    plt.subplot(1,3,1)
    xlab = 'd18O'
    clr = 'gray';
    plt.locator_params(axis = 'x',nbins = 4);
    plt.plot(d18O,depth,color = clr,marker = '.',linestyle = '')    
    plt.xlim([-50,-20])
    plt.ylim([-100,15])
    plt.grid()
    plt.xlabel(xlab)
    plt.locator_params(axis = 'x',nbins = 5);
    plt.plot(bks,depth,'_r',markersize = 20)
    plt.plot(hrs,depth,'_g',markersize = 20)
    

    plt.subplot(1,3,2)
    xlab = 'dD'
    clr = 'red';
    plt.text(-350,-90,date.strftime('%Y%m%d'),rotation = 90)
    plt.locator_params(axis = 'x',nbins = 4);
    plt.plot(d18O,depth,color = clr,marker = '.',linestyle = '')    
    plt.xlim([-380,-150])
    plt.ylim([-100,15])
    plt.grid()
    plt.xlabel(xlab)
    plt.locator_params(axis = 'x',nbins = 2);
    plt.title(title)

    plt.subplot(1,3,3)
    xlab = 'dxs'
    clr = 'blue';
    plt.locator_params(axis = 'x',nbins = 4);
    plt.plot(d18O,depth,color = 'blue',marker = '.',linestyle = '')    
    plt.xlim([-20,10])
    plt.ylim([-100,15])
    plt.grid()
    plt.xlabel(xlab)
    plt.locator_params(axis = 'x',nbins = 2);
    plt.text(-20,-90,date.strftime('%Y%m%d'),rotation = 90)


# useful stuff
#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'ppt' # '\textperthousand'

# main

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019.pkl'
fileWrkngFldr = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/workingFolder'
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

# ***************************************************
# plot individual cores with correlations
# ***************************************************

# make error array for dxs, find correlation with d18O errors

testSize = 0.3;
df_iso['dxs_std'] = df_iso.dD_std+df_iso.d18O_std*8
df_iso.dropna(inplace = True)

m, b, r2 = regressPipeline_dDd18O(sm.add_constant(df_iso.d18O_std),df_iso.dxs_std, testSize, 'dxsStd vs d18Ostd for all EastGRIP Snow Core data', 
                                  d18Osym + ' std (ppt)', 'dxs std (ppt)', np.asarray([0, 0.1]), np.asarray([0, 1.5]))
plt.legend(['scatter','regression'],loc = 'upper left')
plt.savefig(figureLoc+'errorCorrelation_dxsStdVsd18Ostd_EastGRIP_2016-2019.jpg')



## quasi infinite loop here... 

for d in df_iso.date[0]:
    
    for cid in df_iso.coreID[0]:
        
        
        d18O = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].d18O
        dD = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].dD
        dxs = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].dexcess
        bks = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].breaks 
        hrs = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].hoar
        depth = df_iso[(df_iso.date == d)&(df_iso.coreID == cid)].depth
        title = 'EastGRIP profiles for: ' + d.strftime('%Y%m%d') + ' pos ' + str(cid);
        # plot profile, dD, d18O, dxs
        plt.figure()
        plotProfileDaily(d,cid,dD,d18O,dxs,bks,hrs,depth,title)

        
        
        # plot dD vs d18O
        
        # plot dxs vs d18O 
        
        # regress all of these with errors in each variable
        
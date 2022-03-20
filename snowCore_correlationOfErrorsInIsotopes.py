#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:47:18 2022

A small investigation into the correlation of errors in the dD and d18O is performed
from the snow cores at EastGRIP. 

@author: michaeltown
"""

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import scipy.stats as stats


def plotResid(yTrue,yResid,xlabelR,ylabelR,titleR):
    
    fig = plt.figure();
    zeroLine = yTrue*0;
    plt.plot(yTrue,yResid,color = 'blue',marker = 'o',alpha = 0.5,ls = 'None');
    plt.plot(yTrue,zeroLine,'k--')
    plt.xlabel(xlabelR);
    plt.ylabel(ylabelR);
    plt.grid;
    plt.title(titleR);
    plt.show;


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
    plt.plot(xlim,8*xlim,'--',color = 'black',alpha = 0.3)        # for equilibrium slope
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

# useful stuff
#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'ppt' # '\textperthousand'


#******************************************
# main
#******************************************

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);

df_iso.dropna(inplace = True)

testSize = 0.3;         

m, b, r2 = regressPipeline_dDd18O(sm.add_constant(df_iso.d18O_std),df_iso.dD_std, testSize, 'dDstd vs d18Ostd for all EastGRIP Snow Core data', 
                                  d18Osym + ' std (ppt)', dDsym + ' std (ppt)', np.asarray([0, 0.15]), np.asarray([0, 1]))
plt.legend(['scatter','regression','eq water line'],loc = 'lower right')
plt.savefig(figureLoc+'errorCorrelation_dDstdVsd18Ostd_EastGRIP_2016-2019.jpg')

m, b, r2 = regressPipeline_dDd18O(sm.add_constant(df_iso.d18O),df_iso.d18O_std, testSize, 'd18Ostd vs d18O for all EastGRIP Snow Core data', 
                                  d18Osym + ' (ppt)', d18Osym + ' std (ppt)', np.asarray([-50,-20]), np.asarray([0, 0.15]))
plt.legend(['scatter','regression'],loc = 'lower right')
plt.savefig(figureLoc+'errorCorrelation_d18OstdVsd18O_EastGRIP_2016-2019.jpg')


m, b, r2 = regressPipeline_dDd18O(sm.add_constant(df_iso.dD),df_iso.dD_std, testSize, 'dDstd vs dD for all EastGRIP Snow Core data', 
                                  dDsym + ' (ppt)', dDsym + ' std (ppt)', np.asarray([-380,-150]), np.asarray([0, 1]))
plt.legend(['scatter','regression'],loc = 'lower right')
plt.savefig(figureLoc+'errorCorrelation_dDstdVsdD_EastGRIP_2016-2019.jpg')


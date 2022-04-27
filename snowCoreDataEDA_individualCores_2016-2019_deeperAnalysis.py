#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:39:09 2022

This code will take a look at individual profiles, plotting all the data from one position
together. profiles of dD, d18O, dxs, then d18O vs dxs all plotted with each other. 

Winter and summer peaks are identified. 

There is further investigation of  

@author: michaeltown
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from pandas_ods_reader import read_ods
import pickle as pkl


#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
pptsym = 'ppt' # '\textperthousand'

# plot profile 

def plotProfile1(date,spn,spid,iso,bks,hrs,depth,title,xlab,ylab,xlim,ylim):

    #set the breaks/hoar values to something that will appear on the graphs properly
    
    bks = [b*np.mean(iso) if b == 1 else np.nan for b in bks]
    hrs = [h*np.mean(iso) if h == 1 else np.nan for h in hrs]


    
    plt.subplot(1,spn,spid)
    if xlab == 'd18O':
        clr = 'gray';
        plt.text(-48,-90,date.strftime('%Y%m%d'),rotation = 90)
        plt.locator_params(axis = 'x',nbins = 4);


    elif xlab == 'dD':
        clr = 'black';
        plt.text(-380,-90,date.strftime('%Y%m%d'),rotation = 90)
        plt.locator_params(axis = 'x',nbins = 2);

    else:
        clr = 'lightblue';
        plt.text(-20,-90,date.strftime('%Y%m%d'),rotation = 90)
        plt.locator_params(axis = 'x',nbins = 2);

    plt.plot(iso,depth,color = clr,marker = '.',linestyle = '')    
    # plt.plot(iso,depth,color = clr,linewidth = 3)
    plt.grid();

    if spid == 3:
    
        if xlab == 'd18O':
            plt.xlabel(d18Osym+ ' ('+pptsym+')')
        elif xlab == 'dD':
            plt.xlabel(dDsym+ ' ('+pptsym+')')
        else:
            plt.xlabel(xlab)

        plt.title(title + ' in ' + str(date.strftime('%Y')))

    if spid == 1:
        plt.ylabel(ylab)

    if spid > 1:
        plt.tick_params(labelleft=False, left=False)

    plt.plot(bks,depth,'_r',markersize = 20)
    plt.plot(hrs,depth,'_g',markersize = 20)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(rotation = 90)
    
# plot the 2016 data in shaded colors to indicate time.
def plotProfile2(spid,df,clr,title,xlab,ylab,xlim,ylim,y):
            
    dateUnique = pd.to_datetime(df.date.unique());
    numDates = len(dateUnique)
    
    clrScl = np.arange(1,0.1,-0.9/numDates)         # assumes dates are linearly spaced, which they mostly are in 2016
    i = 0;
    for d in dateUnique:
    
        if xlab == 'd18O':
            iso = df[(df.date == d)].d18O;
        elif xlab == 'dD':
            iso = df[(df.date == d)].dD;
        elif xlab == 'd-excess':
            iso = df[(df.date == d)].dexcess;
        else:
            print('I do not recognize that quantity.')
            return


        depth = df[(df.date == d)].depth
        plt.plot(iso,-depth,'o',color = clr,alpha = clrScl[i])
        i += 1
    
    plt.grid();
    
    if xlab == 'd18O':
        plt.xlabel(d18Osym+ ' ('+pptsym+')')
    elif xlab == 'dD':
        plt.xlabel(dDsym+ ' ('+pptsym+')')
    else:
        plt.xlabel(xlab)

    plt.title(title + ' in ' + str(y))
    plt.ylabel(ylab)
    plt.xlim(xlim)
    plt.ylim(ylim)



#*************************************
# main
#*************************************

# This codeis in progress (27 April 2022)


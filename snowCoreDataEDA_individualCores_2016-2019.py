#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:33:58 2022

@author: michaeltown
"""

'''
Here I do initial EDA on each year of the snow cores, including the break points observed in each core.

'''

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

# depth scale calculation from sample number
def depthScaleSampNum(sampNum):
    
    if np.isnan(sampNum) == False:
        sampList = list(np.arange(1,11))+list(np.arange(11,55));
        scaleList = list(np.arange(0.5,11,1.1))+list(np.arange(11.5,107,2.2))
        scaleDict = dict(zip(sampList,scaleList))
        
        return scaleDict[sampNum]
    else:
        # print(sampNum)
        return np.nan

# sample number calculation from depth
def sampNumDepthScale(d):
    
    if np.isnan(d) == False:
        sampList = list(np.arange(1,11))+list(np.arange(11,55));
        scaleList = list(np.arange(0.5,11,1.1))+list(np.arange(11.5,107,2.2))
        scaleDict = dict(zip(np.round(scaleList,1),sampList))
        
        return scaleDict[d]
    else:
        print(sampNum)
        return np.nan


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


# process the breaks/hoar data to apply properly
def brkHrProcess(bh):

    
    breakHoarList = np.asarray([]);

    for el in bh:

        # clean string of comments
        el = ''.join(c for c in el if not c.isalpha())
            

        # find decimals, find dashes insert a range of numbers between dashes, inclusive of 1/2 cm intervals
        if '-' in el:
            nums = el.split('-')
            nums = np.arange(np.floor(float(nums[0])),np.ceil(float(nums[1]))+1)
        elif ('.' in el) & ('-' not in el):
                nums = np.floor(float(el))           # just one break point/hoar for each single event
        else:
            nums = float(el)
        
        breakHoarList = np.append(breakHoarList,nums)
        
    
    return breakHoarList
     


def breaksApply(sn, d, i, bv):
    if sn==ind and d==b:
        return 1
    else:
        return 0
    


fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019.pkl'
fileNameMeta = 'eastGRIP_metaData_2017-2019.pkl'

df_iso = pd.read_pickle(fileLoc+fileNameIso);
df_meta = pd.read_pickle(fileLoc+fileNameMeta);

# make a date column and sort values
df_iso['date'] = pd.to_datetime(dict(zip(['year','month','day'],[df_iso.year,df_iso.month,df_iso.day])))
df_iso.sort_values(by='date', inplace = True)


# for ind in df_meta.index:
#     acc = float(df_meta[df_meta.index == ind].accumulation)    
#     idx = df_iso[df_iso.sampleNames == ind].index

#     depth = np.asarray(df_iso[df_iso.sampleNames == ind].depth)
#     df_iso.loc[idx, 'depthAcc' ] = depth + acc
    
#    for d in depth:
#        df_iso.loc[idx, 'depthAcc'] = d + acc


# make break and hoar columns in the data frame
df_iso['breaks'] = 0;
df_iso['hoar'] = 0;

# insert breaks info
# this is looping but not assigning values to breaks. 
df_breaksTrue = df_meta.loc[df_meta.breaks.notnull()].breaks;    

# construct a list of sample names with breaks, first need rounded numbers from the depth scale
df_iso.depth = np.round(df_iso.depth,1);        # round everything to make comparison easier
depthScale = df_iso.depth.unique();             # pull the current depth scale 
depthScale.sort();


breaksAll = []

for ind in df_breaksTrue.index:
    
    brks = df_breaksTrue[df_breaksTrue.index == ind]
    brkVals = brkHrProcess(list(brks.values)[0]);         # function that manages the breaks list
    brkVals = [np.round(min(depthScale, key = lambda x: abs(x-bv)),1) for bv in brkVals]  # necessary to keep numbers exactly equal to each other when comparing
    
    
    for b in brkVals:
        b = sampNumDepthScale(b);
        if b < 10:
            strB = '0'+str(b)
        else:
            strB = str(b)
            
        breaksAll.append(str(ind)+'_'+strB)

# assign breaks to the df_iso.breaks column
for ind in breaksAll:
    
    if ind in df_iso.index:
        df_iso.loc[df_iso.index == ind,'breaks'] = 1;
    #else:
        #print(ind)              # things that don't get assigned for some reason (ok, some cores don't exist, some cores in the 1st 10 cm are not labeled properly by iceland group)


# insert hoar values
df_hoarTrue = df_meta.loc[df_meta.hoar.notnull()].hoar;

hoarAll = [];

for ind in df_hoarTrue.index:
    
    hrs = df_hoarTrue[df_hoarTrue.index == ind]
    hrVals = brkHrProcess(list(hrs.values)[0]);     # function that manages the hoar list
    hrVals = [np.round(min(depthScale, key = lambda x: abs(x-hv)),1) for hv in hrVals]  # necessary to keep numbers exactly equal to each other when comparing
    
    
    for h in hrVals:
        h = sampNumDepthScale(h);
        if h < 10:
            strH = '0'+str(h)
        else:
            strH = str(h)
            
        hoarAll.append(str(ind)+'_'+strH)

# assign hoar to the df_iso.breaks column
for ind in hoarAll:
    
    if ind in df_iso.index:
        df_iso.loc[df_iso.index == ind,'hoar'] = 1;


# clean and apply the accumulation information to the depth scales
df_meta.loc[df_meta.accumulation.isnull(),'accumulation'] = 0
df_iso['depthAcc'] = df_iso.depth;


#this statement cleans the df_meta data of erroneously duplicated indexes. ~5 rows will be dropped.
duplicateRows = df_meta[df_meta.index.duplicated()]
df_meta = df_meta.drop(index = duplicateRows.index)

# see notes_snowCoreDataMunging.txt for details. ~13 rows will be dropped
duplicateRows = df_iso[df_iso.index.duplicated()]
df_iso = df_iso.drop(index = duplicateRows.index)

# merge the data sets to get the accumulation data in place
df_temp = df_meta.accumulation;
df_iso = df_iso.merge(df_temp,left_on = 'sampleNames',right_index= True,how = 'left')

oldNames = ['accumulation_x'];
newNames = ['accumulation'];
df_iso.rename(columns = dict(zip(oldNames,newNames)),inplace=True);
df_iso = df_iso.drop(columns = 'accumulation_y')

df_iso.loc[df_iso.accumulation.isnull(),'accumulation'] = 0
df_iso['depthAcc'] = df_iso.depth-df_iso.accumulation

    
    

# plot all data in one year at one location as separate subplots 

os.chdir(figureLoc)

coreID = np.arange(1,6);
yearUnique = df_iso.year.unique();

for y in yearUnique:
    
    
    for c in coreID:  
        dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)]    
        

        figO18 = plt.figure()        
        dateUnique = pd.to_datetime(dfTemp.date.unique());
        numDates = len(dateUnique)
        i = 1;
        for d in dateUnique:
            
            iso18O = dfTemp[(dfTemp.date == d)].d18O;
#            depth = dfTemp[(dfTemp.date == d)].depth
            depth = dfTemp[(dfTemp.date == d)].depthAcc
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar
            
            
            if i == 3:
                titleStr = 'individual d18O: pos ' + str(c);
            else:
                titleStr = '';            
            plotProfile1(d,numDates,i,iso18O,brksTemp,hrsTemp,-1*depth,titleStr,'d18O','depth (cm)',[-50,-20],[-100,15])
            i = i + 1;
        plt.show()
        figO18.savefig('./'+str(y)+'snowCoreIndividual_d18O'+str(y)+'_pos_'+str(c)+'.jpg') 
        
        # could do this without two loops if I could use figure handles better

        figD = plt.figure()        
        i = 1;
        for d in dateUnique:
            
            isoD = dfTemp[(dfTemp.date == d)].dD;
#            depth = dfTemp[(dfTemp.date == d)].depth
            depth = dfTemp[(dfTemp.date == d)].depthAcc
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar

            if i == 3:
                titleStr = 'individual dD: pos ' + str(c);
            else:
                titleStr = '';            
            plotProfile1(d,numDates,i,isoD,brksTemp,hrsTemp,-1*depth,titleStr,'dD','depth (cm)',[-380,-150],[-100,15])
            i = i + 1;
        plt.show()
        figD.savefig('./'+str(y)+'/snowCoreIndividual_dD_pos_'+str(y)+'_pos_'+str(c)+'.jpg') 

        figD = plt.figure()        
        i = 1;
        for d in dateUnique:
            iso18O = dfTemp[(dfTemp.date == d)].d18O            
            isoD = dfTemp[(dfTemp.date == d)].dD;
            dexcess = isoD-8*iso18O;
#            depth = dfTemp[(dfTemp.date == d)].depth
            depth = dfTemp[(dfTemp.date == d)].depthAcc
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar


            if i == 3:
                titleStr = 'individual d-excess: pos ' + str(c);
            else:
                titleStr = '';            
            plotProfile1(d,numDates,i,dexcess,brksTemp,hrsTemp,-1*depth,titleStr,'d-excess','depth (cm)',[-5,20],[-100,15])
            i = i + 1;
        plt.show()
        figD.savefig('./'+str(y)+'snowCoreIndividual_dexcess_'+str(y)+'_pos_'+str(c)+'.jpg') 
        
               
# plot the 2016 snowcore data



y = 2016;
for c in coreID[0:2]:  
    dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)]    
#    dexcess = dfTemp.dD-8*dfTemp.d18O
    titleStr = 'individual d18O: pos ' + str(c);
    figO18 = plt.figure()        
    plotProfile2(c,dfTemp,'black',titleStr,'d18O','depth (cm)',[-50,-20],[-100,10],y)
    figO18.savefig('./'+str(y)+'snowCoreIndividual_d18O_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual dD: pos ' + str(c);    
    figdD = plt.figure()        
    plotProfile2(c,dfTemp,'red',titleStr,'dD','depth (cm)',[-400,-200],[-100,10],y)
    figdD.savefig('./'+str(y)+'snowCoreIndividual_dD_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual dexcess: pos ' + str(c);
    figdD = plt.figure()        
    plotProfile2(c,dfTemp,'darkblue',titleStr,'d-excess','depth (cm)',[-8,20],[-100,10],y)
    figdD.savefig('./'+str(y)+'snowCoreIndividual_dexcess_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual d18O: pos ' + str(c);
    figO18 = plt.figure()        
    plotProfile2(c,dfTemp,'black',titleStr,'d18O','depth (cm)',[-50,-20],[-40,10],y)
    figO18.savefig('./'+str(y)+'snowCoreIndividual_d18O_'+str(y)+'_pos_'+str(c)+'(2).jpg')    
    
    titleStr = 'individual dD: pos ' + str(c);
    figdD = plt.figure()        
    plotProfile2(c,dfTemp,'red',titleStr,'dD','depth (cm)',[-400,-200],[-40,10],y)
    figdD.savefig('./'+str(y)+'snowCoreIndividual_dD_'+str(y)+'_pos_'+str(c)+'(2).jpg')    

    titleStr = 'individual dexcess: pos ' + str(c);
    figdD = plt.figure()        
    plotProfile2(c,dfTemp,'darkblue',titleStr,'d-excess','depth (cm)',[-8,20],[-40,10],y)
    figdD.savefig('./'+str(y)+'snowCoreIndividual_dexcess_'+str(y)+'_pos_'+str(c)+'(2).jpg')    


## save the data file with the breaks, hoar, and accumulation info
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/')
dataFileName = 'eastGRIP_SCisoData_2016-2019.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_iso,outfile);
outfile.close();


# plot average of each day (1-5 positions) as separate subplots, with std

# plot all data in one year at one location as a contour plot with x-axis as distance, especially the 
# 2016 data. These are hard to decipher right now.

# plot all data in one year at one location as a contour plot with x-axis as time 

# add compression information to plots above.

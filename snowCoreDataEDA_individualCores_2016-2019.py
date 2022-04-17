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
from scipy.signal import find_peaks
import figureMagic as fm

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

# adjust the depth scale once the accumulation is taken into account
# def adjustDepthScale(d,dn):
    
#     dAdj = np.asarray([]);
    
#     for dep in d:
        
#         dAdj = dn[np.abs(dn - d).argmin()]
        
#     return dAdj


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
     
# apply break values
def breaksApply(sn, d, i, bv):
    if sn==ind and d==b:
        return 1
    else:
        return 0

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/';
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

df_iso.loc[df_iso.accumulation.isnull(),'accumulation'] = 0
df_iso['depthAcc'] = df_iso.depth-df_iso.accumulation

# make this depth scale regular
depthNew = np.append(np.arange(-15,0,1),df_iso.depth.unique())

df_iso['depthAcc_reg'] = df_iso['depthAcc'].apply(lambda x: depthNew[np.abs(depthNew - x).argmin()])

    

# plot all data in one year at one location as separate subplots 

# find peak params
dist = 8; wid = 2;


os.chdir(figureLoc)
coreID = np.arange(1,6);
yearUnique = df_iso.year.unique();

# create peaks column
df_iso['peaks'] = df_iso.breaks*0;
df_iso['peaksMax'] = df_iso.breaks*0;
df_iso['peaksMin'] = df_iso.breaks*0;



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
#            depth = dfTemp[(dfTemp.date == d)].depthAcc
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar
            peaks, _ = find_peaks(iso18O,distance = dist,width = wid)
            troughs, _ = find_peaks(-iso18O,distance = dist,width = wid)
            maxMin = np.append(peaks,troughs)            
            
            # assign peak values to primary dataframe
            #if len(depth[peaks].index) > 0:
            df_iso.loc[depth[maxMin].index,'peaks'] = 1            
            df_iso.loc[depth[peaks].index,'peaksMax'] = 1            
            df_iso.loc[depth[troughs].index,'peaksMin'] = 1            
            
            
            if i == 3:
                titleStr = 'individual d18O: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,iso18O,brksTemp,hrsTemp,-1*depth,titleStr,'d18O','depth (cm)',[-50,-20],[-100,15])
            plt.plot(iso18O[maxMin],-depth[maxMin],'x',color = 'orange')
            i = i + 1;

        plt.show()
        figO18.savefig('./'+str(y)+'/snowCoreIndividual_d18O'+str(y)+'_pos_'+str(c)+'.jpg') 
        
        # could do this without two loops if I could use figure handles better

        figD = plt.figure()        
        i = 1;
        for d in dateUnique:
            
            isoD = dfTemp[(dfTemp.date == d)].dD;
#            depth = dfTemp[(dfTemp.date == d)].depth
#            depth = dfTemp[(dfTemp.date == d)].depthAcc
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar

            if i == 3:
                titleStr = 'individual dD: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,isoD,brksTemp,hrsTemp,-1*depth,titleStr,'dD','depth (cm)',[-380,-150],[-100,15])
            i = i + 1;
        plt.show()
        figD.savefig('./'+str(y)+'/snowCoreIndividual_dD_pos_'+str(y)+'_pos_'+str(c)+'.jpg') 

        figD = plt.figure()        
        i = 1;
        for d in dateUnique:
            dexcess = dfTemp[(dfTemp.date == d)].dexcess;
#            depth = dfTemp[(dfTemp.date == d)].depth
#            depth = dfTemp[(dfTemp.date == d)].depthAcc
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar

            if i == 3:
                titleStr = 'individual d-excess: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,dexcess,brksTemp,hrsTemp,-1*depth,titleStr,'d-excess','depth (cm)',[-5,20],[-100,15])
            i = i + 1;
        plt.show()
        figD.savefig('./'+str(y)+'/snowCoreIndividual_dexcess_'+str(y)+'_pos_'+str(c)+'.jpg') 
        
        figD = plt.figure()        
        i = 1;
        for d in dateUnique:
            dxsln = dfTemp[(dfTemp.date == d)].dxsln
#            depth = dfTemp[(dfTemp.date == d)].depth
#            depth = dfTemp[(dfTemp.date == d)].depthAcc
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar


            if i == 3:
                titleStr = 'individual dxsln: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,dxsln,brksTemp,hrsTemp,-1*depth,titleStr,'dxsln','depth (cm)',[0,35],[-100,15])
            i = i + 1;
        plt.show()
        figD.savefig('./'+str(y)+'/snowCoreIndividual_dxsln_'+str(y)+'_pos_'+str(c)+'.jpg') 
               
# plot the 2016 snowcore data



y = 2016;
for c in coreID[0:2]:  
    dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)]    
#    dexcess = dfTemp.dD-8*dfTemp.d18O
    titleStr = 'individual d18O: pos ' + str(c);
    figO18 = plt.figure()        
    fm.plotProfile2(c,dfTemp,'black',titleStr,'d18O','depth (cm)',[-50,-20],[-100,10],y)
    figO18.savefig('./'+str(y)+'/snowCoreIndividual_d18O_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual dD: pos ' + str(c);    
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'blue',titleStr,'dD','depth (cm)',[-400,-200],[-100,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dD_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual dexcess: pos ' + str(c);
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'lightblue',titleStr,'d-excess','depth (cm)',[-5,20],[-100,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dexcess_'+str(y)+'_pos_'+str(c)+'.jpg')    

    titleStr = 'individual dxsln: pos ' + str(c);
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'deepskyblue',titleStr,'dxsln','depth (cm)',[0,35],[-100,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dxsln_'+str(y)+'_pos_'+str(c)+'.jpg')    


    titleStr = 'individual d18O: pos ' + str(c);
    figO18 = plt.figure()        
    fm.plotProfile2(c,dfTemp,'black',titleStr,'d18O','depth (cm)',[-50,-20],[-40,10],y)
    figO18.savefig('./'+str(y)+'/snowCoreIndividual_d18O_z'+str(y)+'_pos_'+str(c)+'(2).jpg')    
    
    titleStr = 'individual dD: pos ' + str(c);
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'blue',titleStr,'dD','depth (cm)',[-400,-200],[-40,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dD_z'+str(y)+'_pos_'+str(c)+'(2).jpg')    

    titleStr = 'individual dexcess: pos ' + str(c);
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'lightblue',titleStr,'d-excess','depth (cm)',[-5,20],[-40,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dexcess_z'+str(y)+'_pos_'+str(c)+'(2).jpg')    

    titleStr = 'individual dxsln: pos ' + str(c);
    figdD = plt.figure()        
    fm.plotProfile2(c,dfTemp,'deepskyblue',titleStr,'dxsln','depth (cm)',[0,35],[-40,10],y)
    figdD.savefig('./'+str(y)+'/snowCoreIndividual_dxsln_z'+str(y)+'_pos_'+str(c)+'(2).jpg')    


# plot the mean annual profiles after the accumulation correction has been added

# stats on grouped data.
columnsToProcess = ['d18O','dD','dexcess','dxsln']

# round everything onto a regular depth axis using depthAcc.
df_EGRIP_profiles_2016 = df_iso[df_iso.year == 2016].groupby(['depthAcc_reg'])[columnsToProcess].mean()
df_EGRIP_profiles_2016[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso[df_iso.year == 2016].groupby(['depthAcc_reg'])[columnsToProcess].std()
df_EGRIP_profiles_2016[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso[df_iso.year == 2016].groupby(['depthAcc_reg'])[columnsToProcess].max()
df_EGRIP_profiles_2016[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso[df_iso.year == 2016].groupby(['depthAcc_reg'])[columnsToProcess].min()
df_EGRIP_profiles_2016[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso[df_iso.year == 2016].groupby(['depthAcc_reg'])[columnsToProcess].count()

df_EGRIP_profiles_2017 = df_iso[df_iso.year == 2017].groupby(['depthAcc_reg'])[columnsToProcess].mean()
df_EGRIP_profiles_2017[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso[df_iso.year == 2017].groupby(['depthAcc_reg'])[columnsToProcess].std()
df_EGRIP_profiles_2017[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso[df_iso.year == 2017].groupby(['depthAcc_reg'])[columnsToProcess].max()
df_EGRIP_profiles_2017[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso[df_iso.year == 2017].groupby(['depthAcc_reg'])[columnsToProcess].min()
df_EGRIP_profiles_2017[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso[df_iso.year == 2017].groupby(['depthAcc_reg'])[columnsToProcess].count()
# find peaks in 2017 data
peaks, _ = find_peaks(df_EGRIP_profiles_2017.d18O,distance = dist,width = wid)
troughs, _ = find_peaks(-df_EGRIP_profiles_2017.d18O,distance = dist,width = wid)
maxMin = np.append(peaks,troughs) 
df_EGRIP_profiles_2017['peaks'] = df_EGRIP_profiles_2017.index*0;
df_EGRIP_profiles_2017['peaksMax'] = df_EGRIP_profiles_2017.index*0;
df_EGRIP_profiles_2017['peaksMin'] = df_EGRIP_profiles_2017.index*0;
df_EGRIP_profiles_2017.loc[df_EGRIP_profiles_2017.iloc[maxMin].index,'peaks'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2017.loc[df_EGRIP_profiles_2017.iloc[peaks].index,'peaksMax'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2017.loc[df_EGRIP_profiles_2017.iloc[troughs].index,'peaksMin'] = 1   # have to do this because index is heights


df_EGRIP_profiles_2018 = df_iso[df_iso.year == 2018].groupby(['depthAcc_reg'])[columnsToProcess].mean()
df_EGRIP_profiles_2018[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso[df_iso.year == 2018].groupby(['depthAcc_reg'])[columnsToProcess].std()
df_EGRIP_profiles_2018[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso[df_iso.year == 2018].groupby(['depthAcc_reg'])[columnsToProcess].max()
df_EGRIP_profiles_2018[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso[df_iso.year == 2018].groupby(['depthAcc_reg'])[columnsToProcess].min()
df_EGRIP_profiles_2018[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso[df_iso.year == 2018].groupby(['depthAcc_reg'])[columnsToProcess].count()
peaks, _ = find_peaks(df_EGRIP_profiles_2018.d18O,distance = dist,width = wid)
troughs, _ = find_peaks(-df_EGRIP_profiles_2018.d18O,distance = dist,width = wid)
maxMin = np.append(peaks,troughs) 
df_EGRIP_profiles_2018['peaks'] = df_EGRIP_profiles_2018.index*0;
df_EGRIP_profiles_2018['peaksMax'] = df_EGRIP_profiles_2018.index*0;
df_EGRIP_profiles_2018['peaksMin'] = df_EGRIP_profiles_2018.index*0;
df_EGRIP_profiles_2018.loc[df_EGRIP_profiles_2018.iloc[maxMin].index,'peaks'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2018.loc[df_EGRIP_profiles_2018.iloc[peaks].index,'peaksMax'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2018.loc[df_EGRIP_profiles_2018.iloc[troughs].index,'peaksMin'] = 1   # have to do this because index is heights


df_EGRIP_profiles_2019 = df_iso[df_iso.year == 2019].groupby(['depthAcc_reg'])[columnsToProcess].mean()
df_EGRIP_profiles_2019[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso[df_iso.year == 2019].groupby(['depthAcc_reg'])[columnsToProcess].std()
df_EGRIP_profiles_2019[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso[df_iso.year == 2019].groupby(['depthAcc_reg'])[columnsToProcess].max()
df_EGRIP_profiles_2019[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso[df_iso.year == 2019].groupby(['depthAcc_reg'])[columnsToProcess].min()
df_EGRIP_profiles_2019[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso[df_iso.year == 2019].groupby(['depthAcc_reg'])[columnsToProcess].count()
peaks, _ = find_peaks(df_EGRIP_profiles_2019.d18O,distance = dist,width = wid)
troughs, _ = find_peaks(-df_EGRIP_profiles_2019.d18O,distance = dist,width = wid)
maxMin = np.append(peaks,troughs) 
df_EGRIP_profiles_2019['peaks'] = df_EGRIP_profiles_2019.index*0;
df_EGRIP_profiles_2019['peaksMax'] = df_EGRIP_profiles_2019.index*0;
df_EGRIP_profiles_2019['peaksMin'] = df_EGRIP_profiles_2019.index*0;
df_EGRIP_profiles_2019.loc[df_EGRIP_profiles_2019.iloc[maxMin].index,'peaks'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2019.loc[df_EGRIP_profiles_2019.iloc[peaks].index,'peaksMax'] = 1   # have to do this because index is heights
df_EGRIP_profiles_2019.loc[df_EGRIP_profiles_2019.iloc[troughs].index,'peaksMin'] = 1   # have to do this because index is heights


# print max min from all the peaks in table with stats, fill in later


# save the mean annual profiles, not quite sure what to do with the break and hoar information here.
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/')

dataFileName = 'eastGRIP_SCmeanProfileData_2016.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_profiles_2016,outfile);
outfile.close();

dataFileName = 'eastGRIP_SCmeanProfileData_2017.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_profiles_2017,outfile);
outfile.close();

dataFileName = 'eastGRIP_SCmeanProfileData_2018.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_profiles_2018,outfile);
outfile.close();

dataFileName = 'eastGRIP_SCmeanProfileData_2019.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_profiles_2019,outfile);
outfile.close();

fileLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'

lbstd = df_EGRIP_profiles_2016.d18O-df_EGRIP_profiles_2016.d18O_std;
ubstd = df_EGRIP_profiles_2016.d18O+df_EGRIP_profiles_2016.d18O_std;
lbmin = df_EGRIP_profiles_2016.d18O_min;
ubmax = df_EGRIP_profiles_2016.d18O_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2016.d18O,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,15],fileLoc,'prof_d18O_EGRIP2016');
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2016.dD-df_EGRIP_profiles_2016.dD_std;
ubstd = df_EGRIP_profiles_2016.dD+df_EGRIP_profiles_2016.dD_std;
lbmin = df_EGRIP_profiles_2016.dD_min;
ubmax = df_EGRIP_profiles_2016.dD_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2016.dD,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,15],fileLoc,'prof_dD_EGRIP2016');
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2016.dexcess-df_EGRIP_profiles_2016.dexcess_std;
ubstd = df_EGRIP_profiles_2016.dexcess+df_EGRIP_profiles_2016.dexcess_std;
lbmin = df_EGRIP_profiles_2016.dexcess_min;
ubmax = df_EGRIP_profiles_2016.dexcess_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2016.dexcess,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,15],fileLoc,'prof_dexcess_EGRIP2016');
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2016.dxsln-df_EGRIP_profiles_2016.dxsln_std;
ubstd = df_EGRIP_profiles_2016.dxsln+df_EGRIP_profiles_2016.dxsln_std;
lbmin = df_EGRIP_profiles_2016.dxsln_min;
ubmax = df_EGRIP_profiles_2016.dxsln_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2016.dxsln,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,15],fileLoc,'prof_dxsln_EGRIP2016');
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

# plot the profiles for 2017
lbstd = df_EGRIP_profiles_2017.d18O-df_EGRIP_profiles_2017.d18O_std;
ubstd = df_EGRIP_profiles_2017.d18O+df_EGRIP_profiles_2017.d18O_std;
lbmin = df_EGRIP_profiles_2017.d18O_min;
ubmax = df_EGRIP_profiles_2017.d18O_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2017.d18O,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,15],fileLoc,'prof_d18O_EGRIP2017');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].d18O,-df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2017.dD-df_EGRIP_profiles_2017.dD_std;
ubstd = df_EGRIP_profiles_2017.dD+df_EGRIP_profiles_2017.dD_std;
lbmin = df_EGRIP_profiles_2017.dD_min;
ubmax = df_EGRIP_profiles_2017.dD_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2017.dD,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,15],fileLoc,'prof_dD_EGRIP2017');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].dD,-df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2017.dexcess-df_EGRIP_profiles_2017.dexcess_std;
ubstd = df_EGRIP_profiles_2017.dexcess+df_EGRIP_profiles_2017.dexcess_std;
lbmin = df_EGRIP_profiles_2017.dexcess_min;
ubmax = df_EGRIP_profiles_2017.dexcess_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2017.dexcess,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,15],fileLoc,'prof_dexcess_EGRIP2017');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].dexcess,-df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2017.dxsln-df_EGRIP_profiles_2017.dxsln_std;
ubstd = df_EGRIP_profiles_2017.dxsln+df_EGRIP_profiles_2017.dxsln_std;
lbmin = df_EGRIP_profiles_2017.dxsln_min;
ubmax = df_EGRIP_profiles_2017.dxsln_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2017.dxsln,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,15],fileLoc,'prof_dxsln_EGRIP2017');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].dxsln,-df_EGRIP_profiles_2017[df_EGRIP_profiles_2017.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

# plot the profiles for 2018
lbstd = df_EGRIP_profiles_2018.d18O-df_EGRIP_profiles_2018.d18O_std;
ubstd = df_EGRIP_profiles_2018.d18O+df_EGRIP_profiles_2018.d18O_std;
lbmin = df_EGRIP_profiles_2018.d18O_min;
ubmax = df_EGRIP_profiles_2018.d18O_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2018.d18O,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,15],fileLoc,'prof_d18O_EGRIP2018');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].d18O,-df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2018.dD-df_EGRIP_profiles_2018.dD_std;
ubstd = df_EGRIP_profiles_2018.dD+df_EGRIP_profiles_2018.dD_std;
lbmin = df_EGRIP_profiles_2018.dD_min;
ubmax = df_EGRIP_profiles_2018.dD_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2018.dD,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,15],fileLoc,'prof_dD_EGRIP2018');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].dD,-df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2018.dexcess-df_EGRIP_profiles_2018.dexcess_std;
ubstd = df_EGRIP_profiles_2018.dexcess+df_EGRIP_profiles_2018.dexcess_std;
lbmin = df_EGRIP_profiles_2018.dexcess_min;
ubmax = df_EGRIP_profiles_2018.dexcess_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2018.dexcess,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,15],fileLoc,'prof_dexcess_EGRIP2018');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].dexcess,-df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2018.dxsln-df_EGRIP_profiles_2018.dxsln_std;
ubstd = df_EGRIP_profiles_2018.dxsln+df_EGRIP_profiles_2018.dxsln_std;
lbmin = df_EGRIP_profiles_2018.dxsln_min;
ubmax = df_EGRIP_profiles_2018.dxsln_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2018.dxsln,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,15],fileLoc,'prof_dxsln_EGRIP2018');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].dxsln,-df_EGRIP_profiles_2018[df_EGRIP_profiles_2018.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

# plot the profiles for 2019
lbstd = df_EGRIP_profiles_2019.d18O-df_EGRIP_profiles_2019.d18O_std;
ubstd = df_EGRIP_profiles_2019.d18O+df_EGRIP_profiles_2019.d18O_std;
lbmin = df_EGRIP_profiles_2019.d18O_min;
ubmax = df_EGRIP_profiles_2019.d18O_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2019.d18O,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,15],fileLoc,'prof_d18O_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].d18O,-df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2019.dD-df_EGRIP_profiles_2019.dD_std;
ubstd = df_EGRIP_profiles_2019.dD+df_EGRIP_profiles_2019.dD_std;
lbmin = df_EGRIP_profiles_2019.dD_min;
ubmax = df_EGRIP_profiles_2019.dD_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2019.dD,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,15],fileLoc,'prof_dD_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].dD,-df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2019.dexcess-df_EGRIP_profiles_2019.dexcess_std;
ubstd = df_EGRIP_profiles_2019.dexcess+df_EGRIP_profiles_2019.dexcess_std;
lbmin = df_EGRIP_profiles_2019.dexcess_min;
ubmax = df_EGRIP_profiles_2019.dexcess_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2019.dexcess,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,15],fileLoc,'prof_dexcess_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].dexcess,-df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

lbstd = df_EGRIP_profiles_2019.dxsln-df_EGRIP_profiles_2019.dxsln_std;
ubstd = df_EGRIP_profiles_2019.dxsln+df_EGRIP_profiles_2019.dxsln_std;
lbmin = df_EGRIP_profiles_2019.dxsln_min;
ubmax = df_EGRIP_profiles_2019.dxsln_max;
fig1, filename = fm.myDepthFunc(df_EGRIP_profiles_2019.dxsln,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,15],fileLoc,'prof_dxsln_EGRIP2019');
plt.subplot(1,5,(1,4))
plt.plot(df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].dxsln,-df_EGRIP_profiles_2019[df_EGRIP_profiles_2019.peaks == 1].index,'x',color = 'orange')
os.chdir(fileLoc)
fig1.savefig(filename+'.jpg')

## save the data file with the breaks, hoar, and accumulation info
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/')
dataFileName = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_iso,outfile);
outfile.close();


# plot average of each day (1-5 positions) as separate subplots, with std, num
# this groupby gives a multiindex data frame.

columnsToProcess = ['d18O','dD','dexcess','dxsln']

df_iso_pos = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].mean()
df_iso_pos[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].std()
df_iso_pos[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].max()
df_iso_pos[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].min()
df_iso_pos[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_iso.groupby(['depthAcc_reg','date'])[columnsToProcess].count()

df_iso_pos = df_iso_pos.reset_index(level = 0);         # should reset the index to date and leave depthAcc_reg as a column
df_iso_pos['year'] = df_iso_pos.index.year

fileLoc = None;


for y in yearUnique:


    for d 
        lbstd = df_iso_pos.d18O-df_EGRIP_profiles_2016.d18O_std;
        ubstd = df_EGRIP_profiles_2016.d18O+df_EGRIP_profiles_2016.d18O_std;
        lbmin = df_EGRIP_profiles_2016.d18O_min;
        ubmax = df_EGRIP_profiles_2016.d18O_max;
        myDepthFunc(df_EGRIP_profiles_2016.d18O,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 '+d18Osym+' profile',
                    'd18O','depth (cm)',[-50,-20],[-100,2],fileLoc,'prof_d18O_EGRIP2016');


# plot all data in one year at one location as a contour plot with x-axis as distance, especially the 
# 2016 data. These are hard to decipher right now.

# plot all data in one year at one location as a contour plot with x-axis as time 

# add compression information to plots above.

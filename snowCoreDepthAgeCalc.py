#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:13:13 2022

computes the depth age profile as a function of depth for each core
plots the results for a bs test.

@author: michaeltown
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
import datetime as dt
import matplotlib.patches as mpatches
import figureMagic as fm
import EastGRIPprocs as eg

# list the data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
listDir = os.listdir(fileLoc)
os.chdir(fileLoc)
listDir.sort()

df_st = pd.read_pickle('surfaceTransectDataEastGRIP2017-2019.pkl')


# cycle through all the data (year-by-year, core-by-core) and do the substitution of each 
# date value into the data frame
ld = [l for l in listDir if 'depthAge20' in l]
ld.sort()
cols = ['da_depth','da_slope']
df_all = pd.DataFrame()
filenameSlopes = 'eastGRIPdepthAgeSlopes.csv'
coreIDs = np.arange(1,6)

if (os.path.exists(fileLoc+filenameSlopes)):
    os.remove(fileLoc+filenameSlopes)

for f in ld:
    df = pd.read_csv(f)
    df[cols] = np.nan
    df['snowDatedt'] = pd.to_datetime(df.snowDate,format = '%Y%m%d')
    df['sampleDate'] = pd.to_datetime([s[4:-3] for s in df.Sample.iloc[:]],format = '%Y%m%d')
    
#    fig = plt.figure()
    
#    alphas = np.arange(0.2,1,(1-0.2)/(len(df.sampleDate.unique())))
#    ic = 0;
    # finds the individual sample date depth age scale
    for sd in df.sampleDate.unique():
        timeDiff = [-t.days for t in df[df.sampleDate == sd].snowDatedt.diff()]
        depthDiff = df[df.sampleDate == sd].depthAcc_reg.diff()
        df.da_depth = df[df.sampleDate == sd].depthAcc_reg-depthDiff/2
        df.da_slope = timeDiff/depthDiff 
        
        # writes all the slope data to a file for review
        cols2csv = ['Sample','depthAcc_reg','da_depth','da_slope','snowDatedt']
    
            
        
        if os.path.exists(fileLoc+filenameSlopes) == False:
            df[cols2csv].dropna(subset = ['da_depth']).to_csv(filenameSlopes,header = True)
        else:
            df[cols2csv].dropna(subset = ['da_depth']).to_csv(filenameSlopes,mode='a',header = False)
        
        
        # plt.plot(df[df.sampleDate == sd].da_slope,-df[df.sampleDate == sd].da_depth,'.',
        #           color = 'blue',markersize = 10,alpha = alphas[ic],label = str(np.datetime64(sd,'D')))  
        # ic += 1
        # plt.xlabel('change in time/change in depth (days/cm)')
        # plt.ylabel('depth (cm)')
        # plt.xlim([0,30])
        # plt.ylim([-100,10])
        # plt.title('age/depth slopes for core ' +f[-5] + ' in ' + f[-10:-6])
        #        df_all = df_all.append(df)    don't use this yet

#    plt.legend()
#    fm.saveas(fig,figureLoc+'ageDepthSlopes_core'+f[-5]+'_'+f[-10:-6])


years = np.arange(2017,2020)
ic = 0;

for y in years:

    for c in coreIDs:
        fig = plt.figure()
        

        colors = ['red','orange','blue']
        ldTemp = [l for l in ld if str(y)+'p'+str(c) in l]
        for f in ldTemp:
            df = pd.read_csv(f)
            df[cols] = np.nan
            df['snowDatedt'] = pd.to_datetime(df.snowDate,format = '%Y%m%d')
            df['sampleDate'] = pd.to_datetime([s[4:-3] for s in df.Sample.iloc[:]],format = '%Y%m%d')
            
            
            alphas = np.arange(1,0.2,-(1-0.2)/(len(df.sampleDate.unique())))
            ia = 0;
            # finds the individual sample date depth age scale
            for sd in df.sampleDate.unique():
                timeDiff = [-t.days for t in df[df.sampleDate == sd].snowDatedt.diff()]
                depthDiff = df[df.sampleDate == sd].depthAcc_reg.diff()
                df.da_depth = df[df.sampleDate == sd].depthAcc_reg-depthDiff/2
                df.da_slope = timeDiff/depthDiff 
                
                plt.plot(df[df.sampleDate == sd].da_slope,-df[df.sampleDate == sd].da_depth,'.',
                         color = 'blue',markersize = 10,alpha = alphas[ia],label = str(sd)[0:10])  
                ia += 1

        plt.xlabel('change in time/change in depth (days/cm)')
        plt.ylabel('depth (cm)')
        plt.xlim([0,30])
        plt.ylim([-100,10])
    
        plt.title('age/depth slopes for core ' + str(c) + ' in '+ str(y))
        plt.legend()
        fm.saveas(fig,figureLoc+'ageDepthSlopes_core'+str(c)+'_'+str(y))    
    


# plot them all on the same axes but different colors, adapted this for different positions
years = np.arange(2017,2020)
fig = plt.figure()
ic = 0;

for y in years:
    
    for c in coreIDs:
        
        colors = ['red','orange','blue']
        ldTemp = [l for l in ld if str(y)+'p'+str(c) in l]
        for f in ldTemp:
            df = pd.read_csv(f)
            df[cols] = np.nan
            df['snowDatedt'] = pd.to_datetime(df.snowDate,format = '%Y%m%d')
            df['sampleDate'] = pd.to_datetime([s[4:-3] for s in df.Sample.iloc[:]],format = '%Y%m%d')
            
            
            alphas = np.arange(1,0.2,-(1-0.2)/(len(df.sampleDate.unique())))
            ia = 0;
            # finds the individual sample date depth age scale
            for sd in df.sampleDate.unique():
                timeDiff = [-t.days for t in df[df.sampleDate == sd].snowDatedt.diff()]
                depthDiff = df[df.sampleDate == sd].depthAcc_reg.diff()
                df.da_depth = df[df.sampleDate == sd].depthAcc_reg-depthDiff/2
                df.da_slope = timeDiff/depthDiff 
                
                plt.plot(df[df.sampleDate == sd].da_slope,-df[df.sampleDate == sd].da_depth,'.',
                         color = colors[ic],markersize = 10,alpha = alphas[ia])  
                ia += 1
    ic += 1

plt.xlabel('change in time/change in depth (days/cm)')
plt.ylabel('depth (cm)')
plt.xlim([0,30])
plt.ylim([-100,10])

plt.title('age/depth slopes for cores 1-5 2017-2019')
red_patch = mpatches.Patch(color='red', label='2017')
orange_patch = mpatches.Patch(color='orange',label='2018')
blue_patch = mpatches.Patch(color='blue', label='2019')
plt.legend(handles = [red_patch,orange_patch,blue_patch])
fm.saveas(fig,figureLoc+'ageDepthSlopes_core'+f[-5]+'_'+str(y))

# join the depth-age data with apply df_iso

df_da = pd.read_csv(filenameSlopes)
df_da.set_index('Sample',inplace = True)
df_da.drop(columns = 'Unnamed: 0',inplace = True)

df_iso = pd.read_pickle('eastGRIP_SCisoData_2016-2019_acc_peaks.pkl')

df_iso_m = pd.merge(df_iso,df_da,left_index=True,right_index= True,how = 'outer')
df_iso_m.drop(columns = 'depthAcc_reg_y',inplace = True)
df_iso_m.rename(columns = {'depthAcc_reg_x':'depthAcc_reg'},inplace = True)





# build and apply the slopes to each section of a profile

# include visual limits to help interpretation
datesHor1 = [20190101,20180101,20170101,20160101,20150101]
datesHor1 = [str(d) for d in datesHor1]
datesDt1 = [pd.to_datetime(d) for d in datesHor1]
dictHorDates1 = dict(zip(datesHor1,datesDt1))

datesHor2 = [20190731,20180731,20170731,20160731,20150731]
datesHor2 = [str(d) for d in datesHor2]
datesDt2 = [pd.to_datetime(d) for d in datesHor2]
dictHorDates2 = dict(zip(datesHor2,datesDt2))

d18Orange = [-50,-10]
dxsRange = [-10,30]


years = np.arange(2017,2020)
cols = ['d18O','dexcess','depthAcc_reg','snowDatedt']
for y in years:
    for c in coreIDs:
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        
        for d in dates:
            
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                              (df_iso_m.date == d)]
            dfTemp = dfTemp[cols]
            
            dfTemp.snowDatedt = pd.to_datetime(dfTemp.snowDatedt.str.replace('-',''),format = '%Y%m%d')
            # could have done this better, but anyway re-add the first snowDatedt as the date of the core
            dfTemp.loc[dfTemp.index[0],'snowDatedt'] = pd.to_datetime(dfTemp.index[0][4:-3],format = '%Y%m%d')
            
            # cut out the short cores in the data set
            if len(dfTemp) > 32:
                # interpolate the time values across the profile, don't do this over profiles with 
                # all Nans past the first element
                # maybe use resample to get everything on a regular grid.
            
                dfDay = pd.DataFrame(dfTemp['snowDatedt'])
                dfDay = dfDay.set_index(np.arange(len(dfDay.snowDatedt)))
            
                dfDayNotNull = dfDay.dropna()
                ind = dfDayNotNull.index;
                
                # loop around the blocks of know information excluding the last date
    
                for i in np.arange(len(ind)-1):
                    # finds dt between two adjacent times
                    beg =dfDayNotNull.loc[ind[i],'snowDatedt']
                    end = dfDayNotNull.loc[ind[i+1],'snowDatedt']
                    begInd = dfDay[dfDay.index == ind[i]].index[0]
                    endInd = dfDay[dfDay.index == ind[i+1]].index[0]
                    timeDelta = end-beg
                    periodNum = endInd - begInd + 1
                    timeRangeTemp = pd.date_range(beg,end,periods=periodNum)
                    dfDay.iloc[begInd:endInd+1,dfDay.columns.get_loc('snowDatedt')]=timeRangeTemp
                
                # last depth interval interpolation based on the previous dt, the extrapolation
                # this is where to add the final extrapolation from the mean accumulation rate
                if str(dfTemp.iloc[max(dfDay.index),3]) == 'NaT':
                    begInd = endInd
                    endInd = max(dfDay.index)
                    periodNum = endInd-begInd
                    depthInterval_dt = timeDelta/periodNum
                    beg = end
                    end = beg+depthInterval_dt*(periodNum)
                    timeRangeTemp = pd.date_range(beg,end,periods = periodNum)
                    dfDay.iloc[begInd+1:endInd+1,dfDay.columns.get_loc('snowDatedt')]=timeRangeTemp
            
                dfTemp.loc[:,'snowDatedt'] = dfDay.loc[:,'snowDatedt'].values
                
                # make sure to use 'loc' to reassign snowDatedt values to the original dataFrame
                
                #d18O 
                fig, ax = plt.subplots()
                plt.subplot(121)
                plt.plot(dfTemp.d18O,-dfTemp.depthAcc_reg,color = 'black',alpha = 0.7); 
                plt.xlabel('d18o');plt.ylabel('date'); 
                plt.xlim([-50,-20])
                plt.ylim([-100,10])
                plt.title( str(d)[0:10]+ ' p: ' +str(c))
                plt.subplot(122)
                plt.plot(dfTemp.d18O,dfTemp.snowDatedt,color = 'black',alpha = 0.7);
                for di in dictHorDates1:
                    plt.plot(d18Orange,[dictHorDates1[di],dictHorDates1[di]],'--',color = 'black', linewidth = 1)
                for di in dictHorDates2:
                    plt.plot(d18Orange,[dictHorDates2[di],dictHorDates2[di]],'--',color = 'black',linewidth = 3, alpha = 0.3)
                plt.xlim([-50,-20])

                if y == 2019:
                    plt.ylim([pd.to_datetime('20161101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
                elif y == 2018:
                    plt.ylim([pd.to_datetime('20160501',format = '%Y%m%d'),pd.to_datetime('20181101',format = '%Y%m%d')])
                elif y == 2017:
                    plt.ylim([pd.to_datetime('20151001',format = '%Y%m%d'),pd.to_datetime('20171101',format = '%Y%m%d')])
                plt.xlabel('d18o');plt.ylabel('date'); 
                
                # assign the dates to the original df_iso data frame
                df_iso_m.loc[dfTemp.index,'snowDatedt'] = dfTemp.snowDatedt            
             
                fm.saveas(fig,figureLoc+'d18OvsDepth_Age_c'+str(c)+'_'+str(d)[0:10].replace('-',''))
                
                #dxs 
                fig, ax = plt.subplots()
                plt.subplot(121)
                plt.plot(dfTemp.dexcess,-dfTemp.depthAcc_reg); 
                plt.xlabel('dxs');plt.ylabel('date'); 
                plt.xlim(dxsRange)
                plt.ylim([-100,10])
                plt.title( str(d)[0:10]+ ' p: ' +str(c))
                plt.subplot(122)
                plt.plot(dfTemp.dexcess,dfTemp.snowDatedt,color = 'blue',alpha = 0.7);
                for di in dictHorDates1:
                    plt.plot(dxsRange,[dictHorDates1[di],dictHorDates1[di]],'--',color = 'black',linewidth = 1)
                for di in dictHorDates2:
                    plt.plot(dxsRange,[dictHorDates2[di],dictHorDates2[di]],'--',color = 'black',linewidth = 3, alpha = 0.3)
                plt.xlim(dxsRange)
                plt.plot(dfTemp.dexcess,dfTemp.snowDatedt); 
                if y == 2019:
                    plt.ylim([pd.to_datetime('20161101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
                elif y == 2018:
                    plt.ylim([pd.to_datetime('20160501',format = '%Y%m%d'),pd.to_datetime('20181101',format = '%Y%m%d')])
                elif y == 2017:
                    plt.ylim([pd.to_datetime('20151001',format = '%Y%m%d'),pd.to_datetime('20171101',format = '%Y%m%d')])
                plt.xlabel('dxs');plt.ylabel('date'); 
                
                # assign the dates to the original df_iso data frame
                df_iso_m.loc[dfTemp.index,'snowDatedt'] = dfTemp.snowDatedt            
                fm.saveas(fig,figureLoc+'dxsVsDepth_Age_c'+str(c)+'_'+str(d)[0:10].replace('-',''))
            

# plot individual profiles of d18O as a function of time from 2017-2019 on the same 
# vertical time axis

# cycle through the cores comparing c1 2017 to c1 2018 and so forth (arbitrary choice when 
# comparing between years)


for c in coreIDs:
    fig = plt.figure()
    clr = dict(zip([2017,2018,2019],['red','orange','blue']))
    for y in years:
        a = 0;
    
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
        for d in dates:
            
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
    
            if len(dfTemp)>25:          # filter out the shortshort cores
                dfTemp = dfTemp[cols]
                # plot each depth age profile
                plt.plot(dfTemp.d18O,dfTemp.snowDatedt,alpha = alphas[a],color = clr[y],label = str(d)[0:10],linewidth =2)

                a +=1
    
    for d in dictHorDates1:
        plt.plot(d18Orange,[dictHorDates1[d],dictHorDates1[d]],'--',color = 'black',linewidth = 0.5)
    for d in dictHorDates2:
        plt.plot(d18Orange,[dictHorDates2[d],dictHorDates2[d]],'--',color = 'black',linewidth = 3, alpha = 0.3)
        
    red_patch = mpatches.Patch(color='red', label='2017')
    orange_patch = mpatches.Patch(color='orange',label='2018')
    blue_patch = mpatches.Patch(color='blue', label='2019')
    purple_patch = mpatches.Patch(color = 'purple',label = 'sfc trans')
    plt.plot(df_st.d18O,df_st.index,color = 'purple',alpha = 0.5, marker = 's',linestyle = '',markersize = 3)
    plt.legend(handles = [red_patch,orange_patch,blue_patch,purple_patch])
    plt.xlim(d18Orange)
    plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
    plt.xlabel('d18O')
    plt.yticks(rotation = 25)
    plt.ylabel('date')            
    plt.title('d18O profiles from each 2017-2019 at position ' + str(c))
    fm.saveas(fig,figureLoc+'d18OvsAge_c'+str(c)+'_'+str(np.min(years))+'-'+str(np.max(years)))

# plot individual profiles of dxs as a function of time from 2017-2019 on the same 
# vertical time axis

for c in coreIDs:

    fig = plt.figure()
    clr = dict(zip([2017,2018,2019],['red','orange','blue']))
    for y in years:
        a = 0;
    
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
        for d in dates:
            
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
    
            if len(dfTemp)>25:          # filter out the shortshort cores
                dfTemp = dfTemp[cols]
                # plot each depth age profile
                plt.plot(dfTemp.dexcess,dfTemp.snowDatedt,alpha = alphas[a],color = clr[y],label = str(d)[0:10],linewidth =2,linestyle = '--')

                a +=1
    
    for d in dictHorDates1:
        plt.plot(dxsRange,[dictHorDates1[d],dictHorDates1[d]],'--',color = 'black',linewidth = 1)
    for d in dictHorDates2:
        plt.plot(dxsRange,[dictHorDates2[d],dictHorDates2[d]],'--',color = 'black',linewidth = 3, alpha = 0.3)
        
    red_patch = mpatches.Patch(color='red', label='2017')
    orange_patch = mpatches.Patch(color='orange',label='2018')
    blue_patch = mpatches.Patch(color='blue', label='2019')
    purple_patch = mpatches.Patch(color = 'purple',label = 'sfc trans')
    plt.plot(df_st.dexcess,df_st.index,color = 'purple',alpha = 0.5,marker = '>',linestyle = '',markersize = 3)
    plt.legend(handles = [red_patch,orange_patch,blue_patch])
    plt.xlim(dxsRange)
    plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
    plt.xlabel('dxs')
    plt.yticks(rotation = 25)
    plt.ylabel('date')
    plt.title('dxs profiles from each 2017-2019 at position ' + str(c))            
    fm.saveas(fig,figureLoc+'dxsvsAge_c'+str(c)+'_'+str(np.min(years))+'-'+str(np.max(years)))


# plot the d18O individual years with all dates

for c in coreIDs:
    for y in years:
        fig = plt.figure()
        a = 0;
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
    
        for d in dates:
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
            # cut out the short cores in the data set
            if len(dfTemp) > 25:
                
                dfTemp = dfTemp[cols]
                plt.plot(dfTemp.d18O,dfTemp.snowDatedt,alpha = alphas[a],color = clr[y],label = str(d)[0:10],linewidth = 2)

                a +=1

        for d in dictHorDates1:
            plt.plot(d18Orange,[dictHorDates1[d],dictHorDates1[d]],'--',color = 'black',linewidth = 1)
        for d in dictHorDates2:
            plt.plot(d18Orange,[dictHorDates2[d],dictHorDates2[d]],'--',color = 'black',linewidth = 3, alpha = 0.3)
        plt.plot(df_st.d18O,df_st.index,color = 'purple',alpha = 0.5, marker = 's',linestyle = '',markersize = 3,label = 'sfc trans')
        plt.legend()        
        plt.xlim(d18Orange)
        plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
        plt.xlabel('d18O')
        plt.yticks(rotation = 25)
        plt.ylabel('date')
        plt.title('d18O profiles for position ' + str(c))
        fm.saveas(fig,figureLoc+'d18OvsAge_c'+str(c)+'_'+str(y))
            
    
# plot the dxs individual years with all dates
for c in coreIDs:
    for y in years:
        fig = plt.figure()
        a = 0;
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
    
        for d in dates:
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
            # cut out the short cores in the data set
            if len(dfTemp) > 25:
                
                dfTemp = dfTemp[cols]
                plt.plot(dfTemp.dexcess,dfTemp.snowDatedt,alpha = alphas[a],color = clr[y],label = str(d)[0:10],linewidth = 2,linestyle = '--')

                a +=1

        for d in dictHorDates1:
            plt.plot(dxsRange,[dictHorDates1[d],dictHorDates1[d]],'--',color = 'black',linewidth = 1)
        for d in dictHorDates2:
            plt.plot(dxsRange,[dictHorDates2[d],dictHorDates2[d]],alpha = 0.3, linestyle = '--',color = 'black',linewidth = 3)

        plt.plot(df_st.dexcess,df_st.index,color = 'purple',alpha = 0.5, marker = '>',linestyle = '', markersize = 3,label = 'sfc trans')
        plt.legend()
        plt.xlim(dxsRange)
        plt.ylim([pd.to_datetime('20141101',format = '%Y%m%d'),pd.to_datetime('20191101',format = '%Y%m%d')])
        plt.xlabel('dxs')
        plt.yticks(rotation = 25)
        plt.ylabel('date')        
        plt.title('dxs profiles for position ' + str(c))
        fm.saveas(fig,figureLoc+'dxsVsAge_c'+str(c)+'_'+str(y))


# plot the d18O individual profiles as a function of depth

for c in coreIDs:
    for y in years:
        fig = plt.figure()
        a = 0;
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
        for d in dates:
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
            # cut out the short cores in the data set
            if len(dfTemp) > 25:
            
                dfTemp = dfTemp[cols]
                plt.plot(dfTemp.d18O,-dfTemp.depthAcc_reg,alpha = alphas[a],color = clr[y],label = str(d)[0:10])
                a +=1
        
        plt.legend()
        plt.xlim([-50,-20])
        plt.ylim([-100,10])
        plt.xlabel('d18O')
        plt.ylabel('depth (cm)')
        plt.title('d18O vs depth profiles for position ' + str(c))
        fm.saveas(fig,figureLoc+'d18OvsDepth_c'+str(c)+'_'+str(y))
                
# plot the dxs individual profiles as a function of depth
for c in coreIDs:
            
    for y in years:
        fig = plt.figure()
        a = 0;
        dates = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)].date.unique()
        dates.sort()
        alphas = np.arange(1,0.1,-0.9/(len(dates)+1))
        for d in dates:
    
            dfTemp = df_iso_m[(df_iso_m.year == y)&(df_iso_m.coreID == c)&
                                  (df_iso_m.date == d)]
            # cut out the short cores in the data set
            if len(dfTemp) > 25:
            
                dfTemp = dfTemp[cols]
                plt.plot(dfTemp.dexcess,-dfTemp.depthAcc_reg,alpha = alphas[a],color = clr[y],label = str(d)[0:10],linestyle = '--')
                a +=1
        
        plt.legend()
        plt.xlim([-10,30])
        plt.ylim([-100,10])
        plt.xlabel('dexcess')
        plt.ylabel('depth (cm)')
        plt.title('dxs vs depth profiles for position ' + str(c))
        fm.saveas(fig,figureLoc+'dxsVsDepth_c'+str(c)+'_'+str(y))

# save the pickled data file 

df_iso = df_iso_m;
df_iso.to_pickle(fileLoc+'eastGRIP_SCisoData_2016-2019_acc_peaks_t.pkl')


# plot the (age/depth)**-1 value (accumulation rate), as a function of time

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/meteo/'
dfSonicAcc = pd.read_pickle(fileLoc+'accumulationDataSonicEastGRIP2014-2019.pkl')


cols = ['snowDatedt','da_slope']
df_acc = df_iso[cols];
df_acc = df_acc.dropna()

# Sonic_mean is in m/month, so we needed to convert this to cm/day 100/30.

fig = plt.figure()
plt.plot(df_acc.snowDatedt,df_acc.da_slope.apply(lambda x: 1/x),'.',alpha = 0.5,label = 'snow cores')
plt.plot(dfSonicAcc.index,dfSonicAcc.Sonic_mean*100/30,'s',markersize = 4, 
         color = 'black', alpha = 0.5,label = 'sonic')
plt.ylabel('acc rate (cm/day)')
plt.title('EastGRIP accumulation rates')
plt.legend()
fm.saveas(fig,figureLoc+'accumulationRatesSonicSnowcoresEastGRIP2015-2019')

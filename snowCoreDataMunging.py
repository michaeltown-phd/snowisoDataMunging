#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:19:36 2022

@author: michaeltown
"""
'''
prior to running this code, please make sure the iceland data are ingested and pickled properly.


data munging code to read in the snow core isotope data from EastGRIP 
this code will 
1. read in the measured d18O and dD data, split the sample names, then apply and initial depth scale based on sample names
    This first step is from a multi-sheet data set. Filename formats vary slightly from year-to-year.
2. read in the supporting field notes from the relevant year of data collection. the formating of the field notes varies from 
    year-to-year.
3. join the data frames on the sample name
4. plot the initial results and save the figures of the profiles for each year.


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from pandas_ods_reader import read_ods
import pickle as pkl
from scipy.signal import find_peaks
 


#symbols
d18Osym = '$\delta^{18}$O' 
dDsym = '$\delta$D'
plt.rcParams['text.latex.preamble']=[r"\usepackage{wasysym}"]
pptsym = 'ppt' # '\textperthousand'


# other assumptions
dexcessSlope = 8.0

# plots histogram with automatic textbox locations
def myHistFunc(x,ra,clr,xll,xlu,yll,ylu,title,xlab,ylab,fileloc,filename):
    fig1 = plt.figure();
    plt.hist(x,bins = ra,color = clr,alpha = 0.5, density = True);
    plt.xlim([xll,xlu]);
    plt.ylim([yll,ylu]);
    plt.title(title);
    plt.ylabel(ylab);
    plt.xlabel(xlab);    
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll,'n = '+str(len(x)));
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll-(ylu-yll)/10,'mean = '+str(np.round(np.mean(x),2)));
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll-2*(ylu-yll)/10,'stdev = '+str(np.round(np.std(x),2)));
    plt.grid();
    os.chdir(fileloc);
    fig1.savefig(filename+'.jpg')
    
# plots time series with preset axes limits 
def myTimeSeriesFunc(x,y,clr,xll,xul,yll,yul,title,xlab,ylab,fileloc,filename):

    fig1 = plt.figure();
    plt.plot(x,y,clr,alpha = 0.5)
    plt.xlim([xll,xul])
    plt.ylim([yll,yul])
    plt.grid();
    plt.title(title)
    plt.xlabel(xlab)
    plt.xticks(rotation=30)
    plt.ylabel(ylab)
    os.chdir(fileloc)
    fig1.savefig(filename+'.jpg')


# plots depth profiles of isotopes with preset axes limits 
def myDepthFunc(x,d,num,clr,lbstd,ubstd,lbmin,ubmax,title,xlab,ylab,xlim,ylim,fileloc,filename):

    fig1 = plt.figure(figsize=(5,5))
    plt.subplot(1,5,(1,4))
    plt.plot(x,d,clr,linewidth = 3)
    plt.plot(lbstd,d,'k--',alpha = 0.5)
    plt.plot(ubstd,d,'k--',alpha = 0.5)
    plt.plot(lbmin,d,'k-.',alpha = 0.2)
    plt.plot(ubmax,d,'k-.',alpha = 0.2)
    plt.grid();
    plt.title(title)
    
    if xlab == 'd18O':
        plt.xlabel(d18Osym+ ' ( '+pptsym+')')
    elif xlab == 'dD':
        plt.xlabel(dDsym+ ' ( '+pptsym+')')
    else:
        plt.xlabel(xlab)
        
    plt.ylabel(ylab)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.subplot(1,5,5)    
#    plt.barh(d, num, 2, color = 'black',alpha = 0.5)
    plt.plot(num,d,'k--',linewidth = 3, alpha = 0.4)
    plt.title('count profile')
    plt.xlabel('count')
    plt.ylim(ylim)
    plt.grid()    
    
    os.chdir(fileloc)
    fig1.savefig(filename+'.jpg')


def dayMonthStr(dm):
    if dm <10:
        return '0'+str(dm)
    else:
        return str(dm)

def sampleDay(s):
    day = s[1][-2:]
    
    return int(day)

def sampleMonth(s):
    mn = s[1][-4:-2]
    return int(mn)

def sampleYear(s):
    year = s[1][:-4]
    year = int(year);
    
    if year < 2000:
        year = year + 2000;
    
    return year

def sampleSPID(s):
    spid = s[0][-1]
    
    if spid == 'P':
        return 1;
    elif int(spid) in [1,2,3,4,5]:
        return int(spid)
    else:
        print('I do not recognize that core id (i.e. not 1-5): '+ spid);
        return np.nan

def sampleInst(s):
    if len(s) == 4:
        return s[3];
    else:
        return np.nan;

# takes spltSample and finds the sample depth number. strips the unwanted tails off of it.
def sampleDepthNum(s):

    if len(s) in [3,4,5]:
        
        sam = s[len(s)-1];
        
        # this if statement has been superceded for something else in the index processing.
        if ('NN' in sam)|('N' in sam)|('D' in sam)|('?' in sam)|('MW' in sam)|('a' in sam)|('b' in sam)|('N1' in sam):
                
                sam = sam.replace('N1','').replace('N','').replace('D','').replace('?','').replace('(','').replace(')','').replace('MW','').replace('a','').replace('b','');
                
        return int(sam)
    else:
        print('Check this sample depth index: ' + ' '.join(s))
        return np.nan

# the sample depth scale may change based on the year of data collection
def depthScaleSampNum(sampNum):
    
    if np.isnan(sampNum) == False:
        sampList = list(np.arange(1,11))+list(np.arange(11,55));
        scaleList = list(np.arange(0.5,11,1.1))+list(np.arange(11.5,107,2.2))
        scaleDict = dict(zip(sampList,scaleList))
        
        return scaleDict[sampNum]
    else:
        return np.nan

def sampleNameClean(s):

    # remove the strange endings, fix the 2016 naming issue
    sam = s.replace('SP1_16','SP1_2016').replace('SP2_16','SP2_2016').replace('SP_16','SP1_2016').replace('N1','').replace('N','').replace('D','').replace('?','').replace('(','').replace(')','').replace('MW','').replace('a','').replace('b','');

    # how to catch the extra underscore, presumes all the singleton depth indexes in the iceland data are taken care of.
    if sam[-2] == '_':
        sam = sam[:-2];

    return sam
    


## starting with reading and parsing the d18O and dD into a data frame

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/'

# could not read the xlsx file, but better to keep the xlsx file unmodified anyway.
fileNameAWI = 'SP_2016To2019_AWI.ods'
fileNameIceland = 'eastGRIP_SCisoData_2017-2019_iceland.pkl'
dfDict = pd.read_excel(fileLoc+fileNameAWI,sheet_name = None)

sheetNames = list(dfDict.keys());
df_EGRIP_SCiso = dfDict[sheetNames[0]];

# create data frame of all samples
for s in sheetNames[1:]:
    df_EGRIP_SCiso = df_EGRIP_SCiso.append(dfDict[s]);

# rename the columns to match the iceland data set
cols = ['Stdev','Stdev.1'];
colsNew = ['d18O_std','dD_std'];

df_EGRIP_SCiso = df_EGRIP_SCiso.rename(columns = dict(zip(cols,colsNew)))

# finding the unique sample names


# read in the data processed by the iceland facility and append to the AWI-processed data
df_EGRIP_iceland = pd.read_pickle(fileLoc+fileNameIceland);
df_EGRIP_SCiso = df_EGRIP_SCiso.append(df_EGRIP_iceland)

# assign sample name to index, split the sample name into important components and columnate
#df_EGRIP_SCiso.dropna(subset = 'Sample')
df_EGRIP_SCiso.index = df_EGRIP_SCiso.Sample.apply(sampleNameClean);
df_EGRIP_SCiso.drop(['Sample','Comment'],axis=1,inplace = True)


df_EGRIP_SCiso['spltSample'] = df_EGRIP_SCiso.index.str.split('_');
df_EGRIP_SCiso['day'] = df_EGRIP_SCiso.spltSample.apply(sampleDay);
df_EGRIP_SCiso['month'] = df_EGRIP_SCiso.spltSample.apply(sampleMonth);
df_EGRIP_SCiso['year'] = df_EGRIP_SCiso.spltSample.apply(sampleYear);
df_EGRIP_SCiso['coreID'] = df_EGRIP_SCiso.spltSample.apply(sampleSPID);
df_EGRIP_SCiso['sampleDepthNum'] = df_EGRIP_SCiso.spltSample.apply(sampleDepthNum);
df_EGRIP_SCiso['depth'] = df_EGRIP_SCiso.sampleDepthNum.apply(depthScaleSampNum);

# compute dexcess
df_EGRIP_SCiso['dexcess'] = df_EGRIP_SCiso.dD - dexcessSlope*df_EGRIP_SCiso.d18O


# dxsln 
dD_ln = np.log(df_EGRIP_SCiso.dD/1000+1)*1000
d18O_ln = np.log(df_EGRIP_SCiso.d18O/1000+1)*1000
df_EGRIP_SCiso['dxsln'] = dD_ln - ((-2.85*10**-2)*d18O_ln**2+8.47*d18O_ln)


df_EGRIP_SCiso['sampleNames'] = 'SP'+df_EGRIP_SCiso.coreID.map(str)+'_'+df_EGRIP_SCiso.year.map(str)+df_EGRIP_SCiso.month.apply(dayMonthStr)+df_EGRIP_SCiso.day.apply(dayMonthStr)


# stats on grouped data.
columnsToProcess = ['d18O','dD','dexcess','dxsln']

df_EGRIP_profiles_2016 = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2016].groupby(['depth'])[columnsToProcess].mean()
df_EGRIP_profiles_2016[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2016].groupby(['depth'])[columnsToProcess].std()
df_EGRIP_profiles_2016[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2016].groupby(['depth'])[columnsToProcess].max()
df_EGRIP_profiles_2016[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2016].groupby(['depth'])[columnsToProcess].min()
df_EGRIP_profiles_2016[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2016].groupby(['depth'])[columnsToProcess].count()

df_EGRIP_profiles_2017 = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2017].groupby(['depth'])[columnsToProcess].mean()
df_EGRIP_profiles_2017[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2017].groupby(['depth'])[columnsToProcess].std()
df_EGRIP_profiles_2017[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2017].groupby(['depth'])[columnsToProcess].max()
df_EGRIP_profiles_2017[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2017].groupby(['depth'])[columnsToProcess].min()
df_EGRIP_profiles_2017[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2017].groupby(['depth'])[columnsToProcess].count()


df_EGRIP_profiles_2018 = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2018].groupby(['depth'])[columnsToProcess].mean()
df_EGRIP_profiles_2018[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2018].groupby(['depth'])[columnsToProcess].std()
df_EGRIP_profiles_2018[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2018].groupby(['depth'])[columnsToProcess].max()
df_EGRIP_profiles_2018[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2018].groupby(['depth'])[columnsToProcess].min()
df_EGRIP_profiles_2018[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2018].groupby(['depth'])[columnsToProcess].count()


df_EGRIP_profiles_2019 = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2019].groupby(['depth'])[columnsToProcess].mean()
df_EGRIP_profiles_2019[['d18O_std','dD_std','dexcess_std','dxsln_std']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2019].groupby(['depth'])[columnsToProcess].std()
df_EGRIP_profiles_2019[['d18O_max','dD_max','dexcess_max','dxsln_max']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2019].groupby(['depth'])[columnsToProcess].max()
df_EGRIP_profiles_2019[['d18O_min','dD_min','dexcess_min','dxsln_min']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2019].groupby(['depth'])[columnsToProcess].min()
df_EGRIP_profiles_2019[['d18O_num','dD_num','dexcess_num','dxsln_num']] = df_EGRIP_SCiso[df_EGRIP_SCiso.year == 2019].groupby(['depth'])[columnsToProcess].count()


## save the isotope data 
os.chdir('/home/michaeltown/work/projects/snowiso/data/EastGRIP/')
dataFileName = 'eastGRIP_SCisoData_2016-2019.pkl';
outfile = open(dataFileName,'wb');
pkl.dump(df_EGRIP_SCiso,outfile);
outfile.close();


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



# plot the profiles for 2016
fileLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'

lbstd = df_EGRIP_profiles_2016.d18O-df_EGRIP_profiles_2016.d18O_std;
ubstd = df_EGRIP_profiles_2016.d18O+df_EGRIP_profiles_2016.d18O_std;
lbmin = df_EGRIP_profiles_2016.d18O_min;
ubmax = df_EGRIP_profiles_2016.d18O_max;
myDepthFunc(df_EGRIP_profiles_2016.d18O,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],fileLoc,'prof_d18O_EGRIP2016');

lbstd = df_EGRIP_profiles_2016.dD-df_EGRIP_profiles_2016.dD_std;
ubstd = df_EGRIP_profiles_2016.dD+df_EGRIP_profiles_2016.dD_std;
lbmin = df_EGRIP_profiles_2016.dD_min;
ubmax = df_EGRIP_profiles_2016.dD_max;
myDepthFunc(df_EGRIP_profiles_2016.dD,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,2],fileLoc,'prof_dD_EGRIP2016');

lbstd = df_EGRIP_profiles_2016.dexcess-df_EGRIP_profiles_2016.dexcess_std;
ubstd = df_EGRIP_profiles_2016.dexcess+df_EGRIP_profiles_2016.dexcess_std;
lbmin = df_EGRIP_profiles_2016.dexcess_min;
ubmax = df_EGRIP_profiles_2016.dexcess_max;
myDepthFunc(df_EGRIP_profiles_2016.dexcess,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],fileLoc,'prof_dexcess_EGRIP2016');

lbstd = df_EGRIP_profiles_2016.dxsln-df_EGRIP_profiles_2016.dxsln_std;
ubstd = df_EGRIP_profiles_2016.dxsln+df_EGRIP_profiles_2016.dxsln_std;
lbmin = df_EGRIP_profiles_2016.dxsln_min;
ubmax = df_EGRIP_profiles_2016.dxsln_max;
myDepthFunc(df_EGRIP_profiles_2016.dxsln,-df_EGRIP_profiles_2016.index,df_EGRIP_profiles_2016.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2016 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,2],fileLoc,'prof_dxsln_EGRIP2016');


# plot the profiles for 2017
lbstd = df_EGRIP_profiles_2017.d18O-df_EGRIP_profiles_2017.d18O_std;
ubstd = df_EGRIP_profiles_2017.d18O+df_EGRIP_profiles_2017.d18O_std;
lbmin = df_EGRIP_profiles_2017.d18O_min;
ubmax = df_EGRIP_profiles_2017.d18O_max;
myDepthFunc(df_EGRIP_profiles_2017.d18O,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],fileLoc,'prof_d18O_EGRIP2017');

lbstd = df_EGRIP_profiles_2017.dD-df_EGRIP_profiles_2017.dD_std;
ubstd = df_EGRIP_profiles_2017.dD+df_EGRIP_profiles_2017.dD_std;
lbmin = df_EGRIP_profiles_2017.dD_min;
ubmax = df_EGRIP_profiles_2017.dD_max;
myDepthFunc(df_EGRIP_profiles_2017.dD,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,2],fileLoc,'prof_dD_EGRIP2017');

lbstd = df_EGRIP_profiles_2017.dexcess-df_EGRIP_profiles_2017.dexcess_std;
ubstd = df_EGRIP_profiles_2017.dexcess+df_EGRIP_profiles_2017.dexcess_std;
lbmin = df_EGRIP_profiles_2017.dexcess_min;
ubmax = df_EGRIP_profiles_2017.dexcess_max;
myDepthFunc(df_EGRIP_profiles_2017.dexcess,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],fileLoc,'prof_dexcess_EGRIP2017');

lbstd = df_EGRIP_profiles_2017.dxsln-df_EGRIP_profiles_2017.dxsln_std;
ubstd = df_EGRIP_profiles_2017.dxsln+df_EGRIP_profiles_2017.dxsln_std;
lbmin = df_EGRIP_profiles_2017.dxsln_min;
ubmax = df_EGRIP_profiles_2017.dxsln_max;
myDepthFunc(df_EGRIP_profiles_2017.dxsln,-df_EGRIP_profiles_2017.index,df_EGRIP_profiles_2017.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,2],fileLoc,'prof_dxsln_EGRIP2017');


# plot the profiles for 2018
lbstd = df_EGRIP_profiles_2018.d18O-df_EGRIP_profiles_2018.d18O_std;
ubstd = df_EGRIP_profiles_2018.d18O+df_EGRIP_profiles_2018.d18O_std;
lbmin = df_EGRIP_profiles_2018.d18O_min;
ubmax = df_EGRIP_profiles_2018.d18O_max;
myDepthFunc(df_EGRIP_profiles_2018.d18O,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],fileLoc,'prof_d18O_EGRIP2018');

lbstd = df_EGRIP_profiles_2018.dD-df_EGRIP_profiles_2018.dD_std;
ubstd = df_EGRIP_profiles_2018.dD+df_EGRIP_profiles_2018.dD_std;
lbmin = df_EGRIP_profiles_2018.dD_min;
ubmax = df_EGRIP_profiles_2018.dD_max;
myDepthFunc(df_EGRIP_profiles_2018.dD,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,2],fileLoc,'prof_dD_EGRIP2018');

lbstd = df_EGRIP_profiles_2018.dexcess-df_EGRIP_profiles_2018.dexcess_std;
ubstd = df_EGRIP_profiles_2018.dexcess+df_EGRIP_profiles_2018.dexcess_std;
lbmin = df_EGRIP_profiles_2018.dexcess_min;
ubmax = df_EGRIP_profiles_2018.dexcess_max;
myDepthFunc(df_EGRIP_profiles_2018.dexcess,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],fileLoc,'prof_dexcess_EGRIP2018');

lbstd = df_EGRIP_profiles_2018.dxsln-df_EGRIP_profiles_2018.dxsln_std;
ubstd = df_EGRIP_profiles_2018.dxsln+df_EGRIP_profiles_2018.dxsln_std;
lbmin = df_EGRIP_profiles_2018.dxsln_min;
ubmax = df_EGRIP_profiles_2018.dxsln_max;
myDepthFunc(df_EGRIP_profiles_2018.dxsln,-df_EGRIP_profiles_2018.index,df_EGRIP_profiles_2018.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2018 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,2],fileLoc,'prof_dxsln_EGRIP2018');


# plot the profiles for 2019
lbstd = df_EGRIP_profiles_2019.d18O-df_EGRIP_profiles_2019.d18O_std;
ubstd = df_EGRIP_profiles_2019.d18O+df_EGRIP_profiles_2019.d18O_std;
lbmin = df_EGRIP_profiles_2019.d18O_min;
ubmax = df_EGRIP_profiles_2019.d18O_max;
myDepthFunc(df_EGRIP_profiles_2019.d18O,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],fileLoc,'prof_d18O_EGRIP2019');

lbstd = df_EGRIP_profiles_2019.dD-df_EGRIP_profiles_2019.dD_std;
ubstd = df_EGRIP_profiles_2019.dD+df_EGRIP_profiles_2019.dD_std;
lbmin = df_EGRIP_profiles_2019.dD_min;
ubmax = df_EGRIP_profiles_2019.dD_max;
myDepthFunc(df_EGRIP_profiles_2019.dD,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dD_num,'blue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 '+dDsym+' profile',
            'dD','depth (cm)',[-400,-150],[-100,2],fileLoc,'prof_dD_EGRIP2019');

lbstd = df_EGRIP_profiles_2019.dexcess-df_EGRIP_profiles_2019.dexcess_std;
ubstd = df_EGRIP_profiles_2019.dexcess+df_EGRIP_profiles_2019.dexcess_std;
lbmin = df_EGRIP_profiles_2019.dexcess_min;
ubmax = df_EGRIP_profiles_2019.dexcess_max;
myDepthFunc(df_EGRIP_profiles_2019.dexcess,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dexcess_num,'lightblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dexcess profile',
            'dexcess','depth (cm)',[-5,20],[-100,2],fileLoc,'prof_dexcess_EGRIP2019');

lbstd = df_EGRIP_profiles_2019.dxsln-df_EGRIP_profiles_2019.dxsln_std;
ubstd = df_EGRIP_profiles_2019.dxsln+df_EGRIP_profiles_2019.dxsln_std;
lbmin = df_EGRIP_profiles_2019.dxsln_min;
ubmax = df_EGRIP_profiles_2019.dxsln_max;
myDepthFunc(df_EGRIP_profiles_2019.dxsln,-df_EGRIP_profiles_2019.index,df_EGRIP_profiles_2019.dxsln_num,'deepskyblue',lbstd,ubstd,lbmin,ubmax,'EGRIP 2019 dxsln profile',
            'dxsln','depth (cm)',[0,35],[-100,2],fileLoc,'prof_dxsln_EGRIP2019');


'''
## make list of the samples in the data files here
uniqueSampleNames = df_EGRIP_SCiso['sampleNames'].unique()
np.savetxt(r'/home/michaeltown/work/projects/snowiso/data/EastGRIP/sampleNamesEGRIP2016-2019.txt', uniqueSampleNames, fmt='%s')
#pd.DataFrame(uniqueSampleNames).to_csv(r'home/michaeltown/work/projects/snowiso/data/EastGRIP/sampleNamesEGRIP2016-2019.txt', header=None, index=None, sep=' ', mode='a')

## read the 'all' sample files in and compare to measured
dfAllSamples = pd.read_csv('allSamplesEGRRIP2016-2019.txt',sep= '\t')
temp = 'SP'+dfAllSamples.coreID.map(str)+'_'+dfAllSamples.date.map(str)
uniqueAllSamplesNames = temp.unique();

np.savetxt(r'/home/michaeltown/work/projects/snowiso/data/EastGRIP/mallSampleNamesEGRIP2016-2019.txt', uniqueAllSamplesNames, fmt='%s')


missingData = list(set(uniqueAllSamplesNames)-set(uniqueSampleNames))
np.savetxt(r'/home/michaeltown/work/projects/snowiso/data/EastGRIP/missingSampleNamesEGRIP2016-2019.txt', missingData, fmt='%s')
'''
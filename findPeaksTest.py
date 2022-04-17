#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:52:32 2022

testing the findpeaks algorithm on our data from EastGRIP

@author: michaeltown
"""

from scipy.signal import find_peaks
import pandas as pd
import figureMagic as fm
import matplotlib.pyplot as plt
import numpy as np

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/';
figureLoc  ='/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'
#fileNameIso = 'eastGRIP_SCmeanProfileData_2019.pkl'
fileNameIso = 'eastGRIP_SCmeanProfileData_2018.pkl'
#fileNameIso = 'eastGRIP_SCmeanProfileData_2017.pkl'
yearStr = fileNameIso[-8:-4]
df_iso = pd.read_pickle(fileLoc+fileNameIso);

# find peaks, testing width values

widths = [0, 2, 4, 8, 16]

for w in widths:
    peaks, _ = find_peaks(df_iso.d18O,width = w)
    troughs, _ = find_peaks(-df_iso.d18O,width = w)
    maxMin = np.append(peaks,troughs)
    plt.figure()
    plt.plot(df_iso.d18O,-df_iso.index)
    plt.plot(df_iso.iloc[maxMin].d18O,-df_iso.iloc[maxMin].index,'x',color = 'orange')
    plt.title('width value = ' + str(w))    


# find peaks, testing distance values

distance = [1, 2, 4, 8, 16, 32]


for d in distance:
    peaks, _ = find_peaks(df_iso.d18O,distance = d)
    troughs, _ = find_peaks(-df_iso.d18O,distance = d)
    maxMin = np.append(peaks,troughs)
    plt.figure()
    plt.plot(df_iso.d18O,-df_iso.index)
    plt.plot(df_iso.iloc[maxMin].d18O,-df_iso.iloc[maxMin].index,'x',color = 'orange')
    plt.title('year = ' + yearStr + '; dist value = ' + str(d))
    

# find peaks, testing threshold values on d18O

thresh = [1, 2, 4, 8, 16]


for t in thresh:
    peaks, _ = find_peaks(df_iso.d18O,threshold = t)
    troughs, _ = find_peaks(-df_iso.d18O,threshold = t)
    maxMin = np.append(peaks,troughs)
    plt.figure()
    plt.plot(df_iso.d18O,-df_iso.index)
    plt.plot(df_iso.iloc[maxMin].d18O,-df_iso.iloc[maxMin].index,'x',color = 'orange')
    plt.title('year = ' + yearStr + '; thresh value = ' + str(t))


# testing a threshold and distance parameterization
thresh = [1]
distance = [1, 2, 4, 8, 16, 32]


for d in distance:
    peaks, _ = find_peaks(df_iso.d18O,threshold = thresh, distance = d)
    troughs, _ = find_peaks(-df_iso.d18O,threshold = thresh, distance = d)
    maxMin = np.append(peaks,troughs)
    plt.figure()
    plt.plot(df_iso.d18O,-df_iso.index)
    plt.plot(df_iso.iloc[maxMin].d18O,-df_iso.iloc[maxMin].index,'x',color = 'orange')
    plt.title('year = ' + yearStr + '; thresh value = ' + str(thresh) + ', dist = ' + str(d))


# testing a width and distance parameterization
w= [2]
distance = [1, 2, 4, 8, 16, 32]


for d in distance:
    peaks, _ = find_peaks(df_iso.d18O,width = w, distance = d)
    troughs, _ = find_peaks(-df_iso.d18O,width = thresh, distance = d)
    maxMin = np.append(peaks,troughs)
    plt.figure()
    plt.plot(df_iso.d18O,-df_iso.index)
    plt.plot(df_iso.iloc[maxMin].d18O,-df_iso.iloc[maxMin].index,'x',color = 'orange')
    plt.title('year = ' + yearStr + '; width value = ' + str(w) + ', dist = ' + str(d))
        


### move the test to individual profiles

fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);

# peak params
dist = 6; 
wid = None; 
hei = 1;
prom= None;

coreID = np.arange(1,6);
yearUnique = df_iso.year.unique();

for y in np.arange(2017,2020,1):
    
    
    for c in coreID:  
        dfTemp = df_iso[(df_iso.coreID == c)&(df_iso.year==y)]    
        

        figO18 = plt.figure()        
        dateUnique = pd.to_datetime(dfTemp.date.unique());
        numDates = len(dateUnique)
        i = 1;
        for d in dateUnique:
            
            iso18O = dfTemp[(dfTemp.date == d)].d18O;
            depth = dfTemp[(dfTemp.date == d)].depthAcc_reg
            brksTemp = dfTemp[(dfTemp.date == d)].breaks
            hrsTemp = dfTemp[(dfTemp.date == d)].hoar
            
            dfT = pd.concat([iso18O,depth,brksTemp,hrsTemp])
            
            peaks, _ = find_peaks(iso18O, distance = dist, height = hei, width = wid, prominence = prom)
            troughs, _ = find_peaks(-iso18O, distance = dist, height = hei, width = wid, prominence = prom)

            maxMin = np.append(peaks,troughs)            
            
            if i == 3:
                titleStr = 'individual d18O: pos ' + str(c);
            else:
                titleStr = '';            
            fm.plotProfile1(d,numDates,i,iso18O,brksTemp,hrsTemp,-1*depth,titleStr,'d18O','depth (cm)',[-50,-20],[-100,15])
            plt.plot(iso18O[peaks],-depth[peaks],'x',color = 'orange')
            plt.plot(iso18O[troughs],-depth[troughs],'x',color = 'blue')

            i = i + 1;
        plt.show()
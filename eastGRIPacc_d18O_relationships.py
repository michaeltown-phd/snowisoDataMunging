#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:06:08 2022

This code will look into relationships between accumulation and isotopic content. 

You must first run:
    eastGRIPpromiceSnowTemperatures
    eastGRIPaccumulationDataProcessing
    snowCoreProfilesVertStack

You might have to run this as a script in the iPython console because some variables created in the previous
scripts are not recognized when this code is run traditionally.
@author: michaeltown
"""

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import figureMagic as fm
import simpleYork as sy
# functions

def interpolate(xval, df, xcol, ycol):
    return np.interp(xval, df[xcol], df[ycol])


## main

# plots the d18O profiles and temperature all together on the same axes
lbstd = (df_2019.d18O-df_2019.d18O_std)*np.nan;
ubstd = (df_2019.d18O+df_2019.d18O_std)*np.nan;
lbmin = df_2019.d18O_min*np.nan;
ubmax = df_2019.d18O_max*np.nan;
_,ax,_ =fm.myDepthFunc2(df_2019.d18O,-df_2019.index,df_2019.d18O_num,'black',lbstd,ubstd,lbmin,ubmax,'EGRIP 2017-2019 '+d18Osym+' profile',
            'd18O','depth (cm)',[-50,-20],[-100,2],None,'prof_d18O_EGRIP2019');
ax.plot(df_2018.d18O,-df_2018.index-acc2018,color = 'black',linewidth = 3,alpha = 0.5,label='2018')
ax.plot(df_2017.d18O,-df_2017.index-acc2018-acc2017,color = 'black',linewidth = 3,alpha = 0.3,label='2017')
plt.ylim(-200,15)
plt.xlim(-50,-20)
plt.legend(loc='lower left')
ax1 = ax.twiny()
beg = pd.to_datetime('20160101',format = '%Y%m%d')
end = pd.to_datetime('20190725',format = '%Y%m%d')
ax1.plot(dfJoin.AirTemperatureC[(dfJoin.index > beg)&(dfJoin.index < end)],
         dfJoin[(dfJoin.index > beg)&(dfJoin.index < end)].Sonic_cumsum_snow_height-
         np.max(dfJoin[(dfJoin.index > beg)&(dfJoin.index < end)].Sonic_cumsum_snow_height)
         ,'--',color = 'red',linewidth = 3, alpha = 0.5, label = 'Air Temp')
ax1.tick_params(axis='x',labelcolor = 'red',color= 'red')
ax1.set(xlim = [-70,10])
plt.legend(loc='lower right')

# scatter plot of 2019 monthly mean temperature vs 2019 monthly d18O indexed on dates derived from accumulation 

# interpolate df_2019 to monthly acc data, remember that 'depth' is positive until it is plotted
# remember to work backwards from the top for the cumsum to be relateable to the d18O

# range and interpolation for the 2019 core
range1 = 0
range2 = 105
dfJoin['depth'] = -1*(dfJoin.Sonic_cumsum_snow_height.values-np.max(dfJoin.Sonic_cumsum_snow_height))
dfJoinSorted = dfJoin.sort_values('depth')
df_2019['depth'] = df_2019.index

d18O2019core_interp = interpolate(dfJoinSorted[(dfJoinSorted.depth>=range1)&(dfJoinSorted.depth<=range2)].depth.values,
                                  df_2019,'depth','d18O')

# range and interpolation for the 2018 core
range3 = 21
range4 = 133
df_2018['depth'] = df_2018.index+acc2018
d18O2018core_interp = interpolate(dfJoinSorted[(dfJoinSorted.depth>=range3)&(dfJoinSorted.depth<=range4)].depth.values,
                                  df_2018[(df_2018.depth>=range3)&(df_2018.depth<=range4)],'depth','d18O')

# range and interpolation for the 2017 core
range5 = 81
range6 = 193
df_2017['depth'] = df_2017.index+acc2017+acc2018
d18O2017core_interp = interpolate(dfJoinSorted[(dfJoinSorted.depth>=range5)&(dfJoinSorted.depth<=range6)].depth.values,
                                  df_2017[(df_2017.depth>=range5)&(df_2017.depth<=range6)],'depth','d18O')


# use the same reorder index on the air temperature values when plotting. There has got to be a better way!
plt.figure()
plt.plot(dfJoinSorted[(dfJoinSorted.depth>=range1)&(dfJoinSorted.depth<=range2)].AirTemperatureC.values,d18O2019core_interp,
         '.k',label = '2019')
plt.plot(dfJoinSorted[(dfJoinSorted.depth>=range3)&(dfJoinSorted.depth<=range4)].AirTemperatureC.values,d18O2018core_interp,
         '.k',alpha = 0.5,label = '2018',marker = 's')
plt.plot(dfJoinSorted[(dfJoinSorted.depth>=range5)&(dfJoinSorted.depth<=range6)].AirTemperatureC.values,d18O2017core_interp,
         '.k',alpha = 0.3,label = '2017',marker = '^')
plt.legend(loc = 'upper right')
plt.xlabel('monthly mean air temperature (oC)')
plt.ylabel('d18O (per mille)')
plt.xlim([-50,0])
plt.ylim([-50,-20])

# find the july and winter temperatures to see the max and min temps
dfJoin[(dfJoin.index.month == 7)|(dfJoin.index.month == 12)|(dfJoin.index.month == 1)|(dfJoin.index.month == 2)|(dfJoin.index.month == 11)].AirTemperatureC.values
dfJoin[(dfJoin.index.month == 7)|(dfJoin.index.month == 12)|(dfJoin.index.month == 1)|(dfJoin.index.month == 2)|(dfJoin.index.month == 11)].depth


'''
from the dfJoin data set
date           temperature
2016-07-15    -9.437164
2017-01-15   -44.480874
2017-07-15   -11.828253
2018-01-15   -42.839637
2018-07-15   -11.834583
2019-02-15   -42.336637
2019-07-15    -8.492742

date         depth
2016-07-15    140.893315
2017-01-15    110.382552
2017-07-15    102.428328
2018-01-15     65.902111
2018-07-15     45.299345
2019-02-15     19.076064
2019-07-15     11.811417
'''

'''
from the isotope data sets
depthAcc_reg
0      -26.9135
13.7   -38.968734
33.5   -32.684940
53.3   -39.095360
75.3   -34.851990
Name: d18O, dtype: float64

top number for 2018 core
np.mean(df_2018[df_2018.index<0].d18O)
Out[138]: -31.386097700784372

[df_2018[df_2018.peaks==1].index+acc2018,df_2018[df_2018.peaks==1].d18O.values]
[Float64Index([43.887928310566004, 71.387928310566, 95.58792831056601,
               126.38792831056601],
              dtype='float64', name='depthAcc_reg'),
 array([-41.16034615, -34.16939516, -39.12713571, -30.608     ])]
Name: d18O, dtype: float64
add acc2018 = 33.4879


df_2017[df_2017.index<0].d18O
Out[140]: 
depthAcc_reg
-9.0   -30.985000
-8.0   -32.975250
-7.0   -32.463625
-6.0   -31.818500
-5.0   -32.373904
-4.0   -31.699021
-3.0   -31.913589
-2.0   -33.877500
-1.0   -34.148261
Name: d18O, dtype: float64

np.mean(df_2017[df_2017.index<-3].d18O)
Out[141]: -32.052549946581195
[df_2017[df_2017.peaks==1].index+acc2018+acc2017,df_2017[df_2017.peaks==1].d18O.values]
[Float64Index([ 86.6169109538052, 110.9169109538052, 132.9169109538052,
               157.1169109538052, 179.1169109538052],
              dtype='float64', name='depthAcc_reg'),
 array([-31.69902083, -41.16524074, -33.17311628, -37.7523875 ,
        -36.31284091])]
Name: d18O, dtype: float64
add acc2018 = 33.4879
add acc2017 = 57.12898264]
'''


# refined dates and depths with minima and maxima in temperature

dfTempMaxMin = pd.DataFrame()
dates = ['2016-07-15', '2017-01-15','2017-07-15','2018-01-15', '2018-07-15', '2019-02-15', '2019-07-15']

Tarray = [-9.437164, -44.480874,-11.828253,-42.839637,-11.834583,-42.336637,-8.492742]
depthArray = [140.893315,110.382552,102.428328,65.902111,45.299345,19.076064,11.811417]
dfTempMaxMin['dates'] = dates;
dfTempMaxMin.set_index('dates',inplace = True)
dfTempMaxMin['Temp'] = Tarray
dfTempMaxMin['depth'] = depthArray


# how to assign a time value for the sensitivity? we will put it as years in the snow 
d18Ocore2019 = pd.DataFrame()
keysCore2019 = ['2017-07-15','2018-01-15', '2018-07-15', '2019-02-15', '2019-07-15']
d18Ocore2019['dates']=keysCore2019
d18Ocore2019.set_index('dates',inplace = True)
d18Ocore2019['d18O'] = [-34.85199,-39.095360,-32.684940,-38.968734,-26.9135] 

d18Ocore2018 = pd.DataFrame()
keysCore2018 = ['2016-07-15', '2017-01-15','2017-07-15','2018-01-15', '2018-07-15']
d18Ocore2018['dates'] = keysCore2018
d18Ocore2018.set_index('dates',inplace = True)
d18Ocore2018['d18O'] = [-30.608,-39.12713571,-34.16939516, -41.16034615,-31.386097700784372 ]

d18Ocore2017 = pd.DataFrame()
keysCore2017 = ['2016-07-15', '2017-01-15','2017-07-15']
d18Ocore2017['dates'] = keysCore2017
d18Ocore2017.set_index('dates',inplace = True)
d18Ocore2017['d18O'] = [-33.17311628,-41.16524074,-32.052549946581195]


yearsInSnow = [1.5,1,0.5,0]
sensitivity = pd.DataFrame()
sensitivity['yearsInSnow'] = yearsInSnow
sensitivity.set_index('yearsInSnow',inplace = True)
sensitivity['dT_2019'] = dfTempMaxMin.Temp.diff()[3:].values
sensitivity['dT_2018'] = dfTempMaxMin.Temp.diff()[1:5].values
sensitivity['dT_2017'] = np.nan
sensitivity['dT_2017'] = np.append([np.nan,np.nan],dfTempMaxMin.Temp.diff()[1:3])


## need to figure out how to efficiently assign variables to a couple of locations here,
# then make sure that all the code is counting from the same direction in time, here
# I think it is the distant past to the present (e.g. from 2016 to 2019)
# then compute the sensitivities...
sensitivity['deld18O_2019'] = d18Ocore2019.d18O.diff()[1:].values
sensitivity['deld18O_2018'] = d18Ocore2018.d18O.diff()[1:].values
sensitivity['deld18O_2017'] = np.append([np.nan,np.nan],d18Ocore2017.d18O.diff()[1:].values)  # klugey

sensitivity['sensCore2019'] = sensitivity.deld18O_2019/sensitivity.dT_2019
sensitivity['sensCore2018'] = sensitivity.deld18O_2018/sensitivity.dT_2018
sensitivity['sensCore2017'] = sensitivity.deld18O_2017/sensitivity.dT_2017


# stack the sensitivities
d18Ounc = 0.1;  # error in d18O measurement
Tunc = 1;       # error in temperature
sensAll = np.append(sensitivity.sensCore2019.values,sensitivity.sensCore2018.values)
sensAll = np.append(sensAll,sensitivity.sensCore2017.values[2:])
sensAllunc = sensAll*0+(np.abs(0.1/np.mean(dfJoin.AirTemperatureC))+
                      np.abs(np.mean(df_2019.d18O)*1/np.mean(dfJoin.AirTemperatureC)**2))
yins = np.append(sensitivity.index,sensitivity.index)
yins = np.append(yins,sensitivity.index[2:])
yinsUnc = yins*0+0.2;

# regression of slopes vs time
p =sensAll*0       # errors are uncorrelated

# computing the uncertainty for the sensitivity, setting the uncertainty for the years in snow

slope1, yinter1, xinter1, xbar1, ybar1 = sy.calcparams(yins,yinsUnc,sensAll,sensAllunc,p)
slope1unc, yinter1unc = sy.calcunc(yins,yinsUnc,sensAll,sensAllunc,xbar1,ybar1,slope1,p)
mswd = sy.calcmswd(yins,yinsUnc,sensAll,sensAllunc, p, slope1, yinter1)

fig, ax = plt.subplots()
plt.plot(sensitivity.index,sensitivity.sensCore2019,'k',marker = 'o',markersize = 7,label = '2019 core' )
plt.plot(sensitivity.index,sensitivity.sensCore2018,'k',marker = 's',label = '2018 core',alpha = 0.5)
plt.plot(sensitivity.index,sensitivity.sensCore2017,'k',marker = '^',markersize = 7,label = '2017 core',alpha = 0.3)
# regression plot
regX = np.asarray([np.min(yins),np.max(yins)])
plt.plot(regX,slope1*regX+yinter1,'r',alpha = 0.5,linestyle = 'dashed',label = 'regression')
#plt.plot(regX,(slope1-slope1unc)*regX+yinter1-yinter1unc,'r',linewidth = 0.3,alpha = 0.5,linestyle = 'dotted')
#plt.plot(regX,(slope1+slope1unc)*regX+yinter1+yinter1unc,'r',linewidth = 0.3,alpha = 0.5,linestyle = 'dotted')
ax.fill_between(regX,(slope1-slope1unc)*regX+yinter1-yinter1unc,(slope1+slope1unc)*regX+yinter1+yinter1unc,color = 'r',alpha = 0.1)
# formatting`
plt.text(0,0.17,'slope = '+str(np.round(slope1,3))+'+/-'+str(np.round(slope1unc,3)))
plt.text(0,0.13,'inter = '+str(np.round(yinter1,3))+'+/-'+str(np.round(yinter1unc,3)))
plt.xlabel('years in the snow')
plt.ylabel('d18O/T (per mille/oC)')
plt.xlim([-0.2,1.7])
plt.ylim([0.1,0.4])
plt.xticks([0,0.5,1,1.5])
plt.yticks([0.1,0.2,0.3,0.4])
plt.legend(loc = 'upper right')


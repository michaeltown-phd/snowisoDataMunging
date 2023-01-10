#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:42:03 2022

Comparison of calibration on dD, d18, dxs before and after July 15-18, 2022 update of samples from iceland

The Iceland lab might have made a mistake in their calibration of the snowiso data from 2016-2019

@author: michaeltown
"""

import pandas as pd
import numpy as np
import os as os
import simpleYork as sy
import matplotlib.pyplot as plt 

# load the data
fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
figureLoc = '/home/michaeltown/work/projects/snowiso/figures/EastGRIP/'

fileNameOld = 'eastGRIP_SCisoData_2016-2019_acc_peaks_Error.csv'
fileNameRecal = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'
os.chdir(fileLoc) 

# import old file and set index
dfOld = pd.read_csv(fileNameOld)
dfOld.set_index(['Sample'],inplace=True)
dfRecal = pd.read_pickle(fileNameRecal)

# plot the residuals of dD, d18O, dxs with regression information, practice using York test

# must merge the two dfs on the sample names
dfM = pd.merge(dfOld,dfRecal,left_index = True,right_index = True)
keepCols = ['d18O_x','d18O_std_x','dD_x','dD_std_x','d18O_y','d18O_std_y','dD_y','dD_std_y']
dfM = dfM[keepCols]
# clean out data that did not change, clean out NaNs
dfM = dfM[(dfM.d18O_x-dfM.d18O_y != 0)]
dfM = dfM.dropna()

# dD relationships
p =dfM.dD_y*0
slope1, yinter1, xinter1, xbar1, ybar1 = sy.calcparams(dfM.dD_y,dfM.dD_std_y,
                                                       dfM.dD_y-dfM.dD_x,
                                                       dfM.dD_std_y+dfM.dD_std_x,p)
slope1unc, yinter1unc = sy.calcunc(dfM.dD_y,dfM.dD_std_y*0+1,dfM.dD_y-dfM.dD_x,
                                   dfM.dD_std_y*0+2,xbar1,ybar1,slope1,p)
mswd = sy.calcmswd(dfM.dD_y,dfM.dD_std_y,dfM.dD_y-dfM.dD_x,dfM.dD_std_y+dfM.dD_std_x, p, slope1, yinter1)

slope1_dD = slope1
yinter1_dD = yinter1
slope1unc_dD = slope1unc
yinter1unc_dD = yinter1unc


dDresid= dfM.dD_y-dfM.dD_x
fig_dD = plt.figure()
plt.plot(dfM.dD_y,dfM.dD_y-dfM.dD_x,'k.',alpha = 0.3)
plt.xlabel('Recalibrated dD (per mille)');
plt.ylabel('residual dD (per mille)')
plt.plot(dfM.dD_y,slope1_dD*dfM.dD_y+yinter1_dD,'r--')
plt.plot(dfM.dD_y,(slope1_dD-slope1unc_dD)*dfM.dD_y+yinter1_dD+yinter1unc_dD,'r--',linewidth = 0.3)
plt.plot(dfM.dD_y,(slope1_dD+slope1unc_dD)*dfM.dD_y+yinter1_dD-yinter1unc_dD,'r--',linewidth = 0.3)
plt.plot([-400, -160],[0,0],'k--')
plt.text(-250,-3,'slope = ' + str(np.round(slope1_dD,4)))
plt.text(-250,-3.5,'interc = ' + str(np.round(yinter1_dD,4)))
plt.text(-250,-4,'mean = ' + str(np.round(np.mean(dDresid),4)))
fig_dD.savefig(figureLoc+'dDRecalResidualPlot_EastGRIP2016-2019.jpg')

# d18O relationships
p =dfM.dD_y*0
d18O_x_unc = np.mean(dfM.d18O_std_x)+dfM.dD_x*0
d18O_y_unc = np.mean(dfM.d18O_std_y)+dfM.dD_x*0

slope1, yinter1, xinter1, xbar1, ybar1 = sy.calcparams(dfM.d18O_y,dfM.d18O_std_y,
                                                       dfM.d18O_y-dfM.d18O_x,
                                                       d18O_x_unc +d18O_y_unc ,p)
slope1unc, yinter1unc = sy.calcunc(dfM.d18O_y,dfM.d18O_std_y*0+0.1,dfM.d18O_y-dfM.d18O_x,
                                   dfM.d18O_std_y*0+0.2,xbar1,ybar1,slope1,p)
mswd = sy.calcmswd(dfM.d18O_y,d18O_y_unc,dfM.d18O_y-dfM.d18O_x,
                   d18O_y_unc+d18O_x_unc, p, slope1, yinter1)


slope1_d18O = slope1
yinter1_d18O = yinter1
slope1unc_d18O= slope1unc
yinter1unc_d18O= yinter1unc


d18Oresid= dfM.d18O_y-dfM.d18O_x
fig_d18O = plt.figure()
plt.plot(dfM.d18O_y,d18Oresid,'k.', alpha = 0.3)
plt.xlabel('Recalibrated d18O (per mille)')
plt.ylabel('residual d18O (per mille)')
plt.plot(dfM.d18O_y,slope1_d18O*dfM.d18O_y+yinter1_d18O,'r--')
plt.plot(dfM.d18O_y,(slope1_d18O-slope1unc_d18O)*dfM.d18O_y+yinter1_d18O+yinter1unc_d18O,'r--',linewidth = 0.3)
plt.plot(dfM.d18O_y,(slope1_d18O+slope1unc_d18O)*dfM.d18O_y+yinter1_d18O-yinter1unc_d18O,'r--',linewidth = 0.3)
plt.plot([-50, -20],[0,0],'k--')
plt.text(-35,0.2,'slope = ' + str(np.round(slope1_d18O,4)))
plt.text(-35,0.15,'interc = ' + str(np.round(yinter1_d18O,4)))
plt.text(-35,0.10,'mean = ' + str(np.round(np.mean(d18Oresid),4)))
fig_d18O.savefig(figureLoc+'d18ORecalResidualPlot_EastGRIP2016-2019.jpg')

# dxs relationships
p =dfM.dD_y*0
dfM['dxs_x'] = dfM.dD_x-8*dfM.d18O_x
dfM['dxs_std_x'] = np.mean(dfM.dD_std_x)-8*d18O_x_unc
dfM['dxs_y'] = dfM.dD_y-8*dfM.d18O_y
dfM['dxs_std_y'] = np.mean(dfM.dD_std_x)-8*d18O_y_unc

slope1, yinter1, xinter1, xbar1, ybar1 = sy.calcparams(dfM.dxs_y,dfM.dxs_std_y,
                                                       dfM.dxs_y-dfM.dxs_x,
                                                       dfM.dxs_std_y+dfM.dxs_std_x,p)
slope1unc, yinter1unc = sy.calcunc(dfM.dxs_y,dfM.dxs_std_y*0+1.8,dfM.dxs_y-dfM.dxs_x,
                                   dfM.dxs_std_y*0+3.6,xbar1,ybar1,slope1,p)
mswd = sy.calcmswd(dfM.dxs_y,dfM.dxs_std_y,dfM.dxs_y-dfM.dxs_x,
                   dfM.dxs_std_y+dfM.dxs_std_x, p, slope1, yinter1)

slope1_dxs = slope1
yinter1_dxs = yinter1
slope1unc_dxs= slope1unc
yinter1unc_dxs= yinter1unc


dxsresid= dfM.dxs_y-dfM.dxs_x
fig_dxs = plt.figure()
plt.plot(dfM.dxs_y,dxsresid,'b.', alpha = 0.5)
plt.xlabel('Recalibrated dxs (per mille)');
plt.ylabel('residual dxs (per mille)')
plt.plot(dfM.dxs_y,slope1_dxs*dfM.dxs_y+yinter1_dxs,'r--')
plt.plot(dfM.dxs_y,(slope1_dxs-slope1unc_dxs)*dfM.dxs_y+yinter1_dxs-yinter1unc_dxs,'r--',linewidth = 0.3)
plt.plot(dfM.dxs_y,(slope1_dxs+slope1unc_dxs)*dfM.dxs_y+yinter1_dxs+yinter1unc_dxs,'r--',linewidth = 0.3)
plt.plot([-10, 20],[0,0],'k--')
plt.text(-5,-2,'slope = ' + str(np.round(slope1_dxs,4)))
plt.text(-5,-2.5,'interc = ' + str(np.round(yinter1_dxs,4)))
plt.text(-5,-3,'mean = ' + str(np.round(np.mean(dxsresid),4)))
fig_dxs.savefig(figureLoc+'dxsRecalResidualPlot_EastGRIP2016-2019.jpg')

# save old data as pkl file
# save new 
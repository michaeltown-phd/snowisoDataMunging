#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:57:45 2022

script to load, flatten, and save data files for submission to pangaea

@author: michaeltown
"""


import pandas as pd

# file locations
fileLocOG = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileLocPang = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/pangaeaSubmission/'

df_md = pd.read_pickle(fileLocOG+'eastGRIP_SCisoData_2017-2019_dindex_accExt_t2.pkl')
df_cd = pd.read_pickle(fileLocOG+'eastGRIP_SCisoData_2017-2019_dindex_accExt_t3.pkl')
df_mt = pd.read_pickle(fileLocOG+'eastGRIP_SCisoData_2017-2019_tindex_accExt_t2.pkl')
df_ct = pd.read_pickle(fileLocOG+'eastGRIP_SCisoData_2017-2019_tindex_accExt_t3.pkl')

df_mdr = df_md.reset_index()
df_cdr = df_cd.reset_index()
df_mtr = df_mt.reset_index()
df_ctr = df_ct.reset_index()

cols = df_mdr.columns
colNewDict = dict(zip(cols,['depth (cm)', 'coreID', 'dateOfExtraction', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)', 'dD_std (per mille)',
       'dexcess (per mille)', 'dxsln (per mille)', 'ageDepth']))
df_mdr.rename(columns = colNewDict,inplace = True)

cols = df_cdr.columns
colNewDict = dict(zip(cols,['depth (cm)', 'coreID', 'dateOfExtraction', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)', 'dD_std (per mille)',
       'dexcess (per mille)', 'dxsln (per mille)', 'ageDepth']))
df_cdr.rename(columns = colNewDict,inplace = True)

cols = df_mtr.columns
colNewDict = dict(zip(cols,['ageDepth', 'coreID', 'dateOfExtraction', 'depth (cm)', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)',
       'dD_std (per mille)', 'dexcess (per mille)', 'dxsln (per mille)']))
df_mtr.rename(columns = colNewDict,inplace = True)

cols = df_ctr.columns
colNewDict = dict(zip(cols,['ageDepth', 'coreID', 'dateOfExtraction', 'depth (cm)', 'd18O (per mille)', 'd18O_std (per mille)', 'dD (per mille)',
       'dD_std (per mille)', 'dexcess (per mille)', 'dxsln (per mille)']))
df_ctr.rename(columns = colNewDict,inplace = True)


print(df_mdr.columns)
print(df_mtr.columns)
print(df_cdr.columns)
print(df_ctr.columns)

df_mdr.to_csv(fileLocPang+'eastGRIP_SCisoData_2017-2019_depthIndex_meanAccModelBot.csv')
df_mtr.to_csv(fileLocPang+'eastGRIP_SCisoData_2017-2019_timeIndex_meanAccModelBot.csv')
df_cdr.to_csv(fileLocPang+'eastGRIP_SCisoData_2017-2019_depthIndex_manualConstModelBot.csv')
df_ctr.to_csv(fileLocPang+'eastGRIP_SCisoData_2017-2019_timeIndex_manualConstModelBot.csv')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:42:05 2022

This script exports snowiso data into a csv file

@author: michaeltown
"""

import pandas as pd
import numpy as np
import pickle as pkl

fileLoc = '/home/michaeltown/work/projects/snowiso/data/EastGRIP/isotopes/'
fileNameIso = 'eastGRIP_SCisoData_2016-2019_acc_peaks.pkl'
df_iso = pd.read_pickle(fileLoc+fileNameIso);

df_iso.to_csv(fileLoc+fileNameIso[:-3]+'csv')


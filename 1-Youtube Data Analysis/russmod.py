# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:58:14 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
df = pd.read_csv('RUSSMOD.csv',encoding='latin1')
df.head()
df_summary=df.describe(include="all")
df_summary.drop(['video_error_or_removed'], axis=1, inplace=True)
df.isnull().sum().sum()
df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)
df_summary.drop(['thumbnail_link'], axis=1, inplace=True)

df.dropna(how='any',inplace=True)

df.isnull().sum().sum()
df_summary=df.describe(include="all")
df_summary

df.head(10)
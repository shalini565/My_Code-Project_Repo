# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:22:05 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.rcParams['figure.figsize'] = (10, 10)

df = pd.read_csv('KRvideos.csv',encoding='latin1')
df.head()
df_categories = pd.read_json('KR_category_id.json')

# Map Category IDs using the supporting file: US_category_id.json
categories = {int(category['id']): category['snippet']['title'] for category in df_categories['items']}


# Category ID will be used to assign categories later, it is not a numeric variable.
df.category_id = df.category_id.astype('category')

# Transform trending_date to datetime date format
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m').dt.date
df.trending_date.value_counts().sort_index(inplace=True)
df.head()

# Dataset is sorted by trending_date
pd.Index(df.trending_date).is_monotonic

# Transforming publish_time to datetime
publish_time = pd.to_datetime(df.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')

# Create Variable publish_date
df['publish_date'] = publish_time.dt.date

# Drop publish_time
df.drop('publish_time',axis=1,inplace=True)

# Create New Variable Counting Days to Achieving Trending Status
df['days_to_trending'] = (df.trending_date - df.publish_date).dt.days
df['days_to_trending']

#drop col most values are missing
df_summary=df.describe(include="all")
df_summary
df_summary.drop(['video_error_or_removed'], axis=1, inplace=True)

df.isnull().sum().sum()
df.dropna(how='any',inplace=True)

df.dtypes
df.isnull().sum().sum()
df.describe()

dcategory={1:"FILM AND ANIMATION",2:"AUTOS AND VEHICLE",10:"MUSIC",15:"PETS & ANIMALS",17:"SPORTS",18:"SHORT MOVIES",19:"TRAVEL & EVENTS",
           20:"GAMING",21:"VIDEOBLOGGING",22:"PEOPLE & BLOGS",23:"COMEDY",24:"ENTERTAINMENT",
           25:"NEWS & POLITICS",26:"HOWTO & STYLE",27:"EDUCATION",28:"SCIENCE & TECHNOLOGY",
           29:"TITLE NONPROFITS & ACTIVISM",30:"MOVIES",31:"ANIME/ANIMATION",
           32:"ACTION/ADVENTURE",33:"CLASSICS",34:"COMEDY",35:"DOCUMENTARY",
           36:"DRAMA",37:"FAMILY",38:"FOREIGN",39:"HORROR",40:"SCI-FI/FANTASY",
           41:"THRILLER",42:"SPORTS",43:"SHOWS",44:"TRAILERS"}

df['category_id'].replace(dcategory,inplace=True)
df['category_id'].describe()
df.head(20)
df.dtypes
df.to_csv('koreamap.csv')
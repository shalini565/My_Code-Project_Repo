# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:38:53 2019

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

df = pd.read_csv('USvideos.csv',encoding='latin1')
df.head()
df_categories = pd.read_json('US_category_id.json')

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
#df.days_to_trending.describe(percentiles=[.05,.25,.5,.75,.95])

#Create Meaningful Data Frame Index
df.set_index(['trending_date','video_id'],inplace=True)
df.head()

#dislike percentage
df['dislike_percentage'] = df['dislikes'] / (df['dislikes'] + df['likes'])
df.dislike_percentage.describe(percentiles=[.05,.25,.5,.75,.95])

# how should we interpret 'video_error_or_removed' == True ?
print(df[df.video_error_or_removed])
df = df[~df.video_error_or_removed]


# Video Level Stats Using First Occurence Values
video_level = df.groupby(level=1).first()
video_level['freq'] = df['title'].groupby(level=1).count()
video_level['category'] = [categories[cid] for cid in video_level.category_id]
video_level.drop('category_id',axis=1,inplace=True)
video_level.sort_values(by=['views'],ascending=False,inplace=True)
video_level['views_ratio'] = df['views'].groupby(level=1).last() / video_level.views
views_min_dt = pd.Series([t[0] for t in df['views'].groupby(level=1).idxmin()],index=video_level.index)
video_level['views_min_dt'] = views_min_dt
video_level.head(10)

# Get Metadata Information
df.info()

video_level.describe(percentiles=[.05,.25,.5,.75,.95])

#Are Video Trending Days Consecutive?
tmp = video_level[['freq','days_to_trending']]
days_to_trending_max = df.groupby(level=1)[['days_to_trending']].max()
tmp = tmp.join(days_to_trending_max,how='left',rsuffix='_max')
tmp['test'] = tmp.days_to_trending_max - tmp.days_to_trending + 1
print('{:.2%}'.format(sum([a==b for a,b in zip(tmp.freq,tmp.test)]) / len(tmp.index)))
tmp[tmp.test != tmp.freq].head()

#Most Infuential Creators
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
tmp = df.channel_title.value_counts()[:25]
_ = sns.barplot(y=tmp.index,x=tmp)

df.dtypes
df.shape
df.describe()

df_summary=df.describe(include="all")
df_summary



#unique value: counts of specific column
df['category_id'].value_counts()

#selected column
colwecare=['title','views','likes','dislikes','comment_count']
df[colwecare].sample(5)

sns.heatmap(df_summary.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#counts of missing data
for c in df_summary.columns:
    print(c,np.sum(df_summary[c].isnull()))
    
#replace missing data
df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)

df.isnull().sum().sum()

#drop col most values are missing
df_summary.drop(['thumbnail_link'], axis=1, inplace=True)
#df.dropna(how='any',inplace=True)

sns.heatmap(df_summary.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.info()

#df['publish_time']=pd.to_datetime(df['publish_time'],format='%Y-%m-%dT%H:%M:%S')

df[['views','likes','dislikes','comment_count']].head()

#feature scaling
df['views']=df['views']/df['views'].max()
df[['views','likes','dislikes','comment_count']].head()

#min-max
df['likes']=(df['likes']-df['likes'].min()) / (df['likes'].max() - df['likes'].min())
df[['views','likes','dislikes','comment_count']].head()

#z-score
#df['likes']=(df['likes']-df['likes'].mean()) / df['likes'].std()
df['dislikes']=(df['dislikes']-df['dislikes'].mean()) / df['dislikes'].std()
df['comment_count']=(df['comment_count']-df['comment_count'].mean()) / df['comment_count'].std()
df[['views','likes','dislikes','comment_count']].head()

#binning convert likes 
binwidth = int(max(df['likes'])-min(df['likes']))/4
binwidth

bins = np.arange(min(df['likes']),max(df['likes']),binwidth)
gp_names = ['low','medium','high']
df['likes-bin']=pd.cut(df['likes'],bins,labels=gp_names)
df['likes-bin']


df['likes-bin'].to_csv("likecat.csv")

#binning convert dislikes 
binwidth1 = int(max(df['dislikes'])-min(df['dislikes']))/3
binwidth1
bins1 = np.arange(min(df['dislikes']),max(df['dislikes']),binwidth1)
gpnames = ['low','medium','high']
df['dislikes-bin']=pd.cut(df['dislikes'],bins1,labels=gpnames)
df['dislikes-bin']

df.describe(percentiles=[.05,.25,.5,.75,.95]).round(1)


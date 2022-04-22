# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:08:50 2019

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

df1 = pd.read_csv('RUvideos.csv',encoding='latin1')
df1.head()
df_categories = pd.read_json('RU_category_id.json')

# Map Category IDs using the supporting file: RU_category_id.json
categories = {int(category['id']): category['snippet']['title'] for category in df_categories['items']}


# Category ID will be used to assign categories later, it is not a numeric variable.
df1.category_id = df1.category_id.astype('category')

# Transform trending_date to datetime date format
df1['trending_date'] = pd.to_datetime(df1['trending_date'], format='%y.%d.%m').dt.date
df1.trending_date.value_counts().sort_index(inplace=True)
df1.head()

# Dataset is sorted by trending_date
pd.Index(df1.trending_date).is_monotonic

# Transforming publish_time to datetime
publish_time = pd.to_datetime(df1.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')

# Create Variable publish_date
df1['publish_date'] = publish_time.dt.date

# Drop publish_time
df1.drop('publish_time',axis=1,inplace=True)

# Create New Variable Counting Days to Achieving Trending Status
df1['days_to_trending'] = (df1.trending_date - df1.publish_date).dt.days
df1['days_to_trending']
#df.days_to_trending.describe(percentiles=[.05,.25,.5,.75,.95])

#Create Meaningful Data Frame Index
df1.set_index(['trending_date','video_id'],inplace=True)
df1.head()

#dislike percentage
df1['dislike_percentage'] = df1['dislikes'] / (df1['dislikes'] + df1['likes'])
df1.dislike_percentage.describe(percentiles=[.05,.25,.5,.75,.95])

# how should we interpret 'video_error_or_removed' == True ?
print(df1[df1.video_error_or_removed])
df1 = df1[~df1.video_error_or_removed]

df1[df1.video_error_or_removed].describe()

#drop col most values are missing
df_summary=df1.describe(include="all")
df_summary
df_summary.drop(['video_error_or_removed'], axis=1, inplace=True)

#drop column
#df_summary.drop(['thumbnail_link'], axis=1, inplace=True)

df1.dtypes

# Video Level Stats Using First Occurence Values
video_level = df1.groupby(level=1).first()
video_level['freq'] = df1['title'].groupby(level=1).count()
video_level['category'] = [categories[cid] for cid in video_level.category_id]
#video_level.drop('category_id',axis=1,inplace=True)
video_level.sort_values(by=['views'],ascending=False,inplace=True)
video_level['views_ratio'] = df1['views'].groupby(level=1).last() / video_level.views
views_min_dt = pd.Series([t[0] for t in df1['views'].groupby(level=1).idxmin()],index=video_level.index)
video_level['views_min_dt'] = views_min_dt
video_level.head(10)

# Get Metadata Information
df1.info()

video_level.describe(percentiles=[.05,.25,.5,.75,.95])

#Are Video Trending Days Consecutive?
tmp = video_level[['freq','days_to_trending']]
days_to_trending_max = df1.groupby(level=1)[['days_to_trending']].max()
tmp = tmp.join(days_to_trending_max,how='left',rsuffix='_max')
tmp['test'] = tmp.days_to_trending_max - tmp.days_to_trending + 1
print('{:.2%}'.format(sum([a==b for a,b in zip(tmp.freq,tmp.test)]) / len(tmp.index)))
tmp[tmp.test != tmp.freq].head()

#Most Infuential Creators
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
tmp = df1.channel_title.value_counts()[:25]
_ = sns.barplot(y=tmp.index,x=tmp)

df1.dtypes
df1.shape
df1.describe()

#Video category distribution
sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
sns_ax = sns.countplot([categories[i] for i in df1.category_id])
_, labels = plt.xticks()
_ = sns_ax.set_xticklabels(labels, rotation=60)

#time variation test
table = pd.pivot_table(df1, index=df1.index.labels[0])
table.index = df1.index.levels[0]
_ = table[['likes','dislikes','comment_count']].plot()
_ = table[['views']].plot()
_ = table[['comments_disabled','ratings_disabled','video_error_or_removed']].plot()

#likes and dislikes
sns_ax = sns.distplot(np.nan_to_num(df1.sample(1000).dislike_percentage),bins='fd')  # Notice Sampling: EDA Principle 3
_ = sns_ax.set_title('Distribution of Dislike Percentage')


#unique value: counts of specific column
df1['category_id'].value_counts()

#selected column
colwecare=['title','views','likes','dislikes','comment_count']
df1[colwecare].sample(5)

sns.heatmap(df_summary.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#counts of missing data
for c in df_summary.columns:
    print(c,np.sum(df_summary[c].isnull()))
    
#replace missing data
df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)

df1.isnull().sum().sum()

df1.to_csv('RUSSMOD.csv')

#drop col most values are missing
df_summary.drop(['thumbnail_link'], axis=1, inplace=True)
df1.dropna(how='any',inplace=True)

sns.heatmap(df_summary.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df1.info()

#df['publish_time']=pd.to_datetime(df['publish_time'],format='%Y-%m-%dT%H:%M:%S')

df1[['views','likes','dislikes','comment_count']].head()

#feature scaling
df1['views']=df1['views']/df1['views'].max()
df1[['views','likes','dislikes','comment_count']].head()

#min-max
df1['likes']=(df1['likes']-df1['likes'].min()) / (df1['likes'].max() - df1['likes'].min())
df1[['views','likes','dislikes','comment_count']].head()

#z-score
#df['likes']=(df['likes']-df['likes'].mean()) / df['likes'].std()
df1['dislikes']=(df1['dislikes']-df1['dislikes'].mean()) / df1['dislikes'].std()
df1['comment_count']=(df1['comment_count']-df1['comment_count'].mean()) / df1['comment_count'].std()
df1[['views','likes','dislikes','comment_count']].head()

#binning convert likes 
binwidth = int(max(df1['likes'])-min(df1['likes']))/4
binwidth

bins = np.arange(min(df1['likes']),max(df1['likes']),binwidth)
gp_names = ['low','medium','high']
df1['likes-bin']=pd.cut(df1['likes'],bins,labels=gp_names)
df1['likes-bin']


df1['likes-bin'].to_csv("likecat.csv")

#binning convert dislikes 
binwidth1 = int(max(df1['dislikes'])-min(df1['dislikes']))/3
binwidth1
bins1 = np.arange(min(df1['dislikes']),max(df1['dislikes']),binwidth1)
gpnames = ['low','medium','high']
df1['dislikes-bin']=pd.cut(df1['dislikes'],bins1,labels=gpnames)
df1['dislikes-bin']

df1.describe(percentiles=[.05,.25,.5,.75,.95]).round(1)


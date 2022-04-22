# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 07:20:00 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('USMOD.csv',encoding='latin1')
df.head()
df_summary=df.describe(include="all")
df_summary
df_summary.drop(['video_error_or_removed'], axis=1, inplace=True)
df.isnull().sum().sum()
df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)
df_summary.drop(['thumbnail_link'], axis=1, inplace=True)

df.dropna(how='any',inplace=True)

df.isnull().sum().sum()
df.describe()
df.drop(['publish_date'],axis=1,inplace=True)
df.drop(['comments_disabled'],axis=1,inplace=True)
df.dtypes

df.drop(['comment_count'],axis=1,inplace=True)
df.drop(['description'],axis=1,inplace=True)
df.drop(['thumbnail_link'],axis=1,inplace=True)
df.drop(['tags'],axis=1,inplace=True)
df.drop(['video_error_or_removed'],axis=1,inplace=True)
df['channel_title'].value_counts()

df.drop(['trending_date'],axis=1,inplace=True)
df.drop(['video_id'],axis=1,inplace=True)
df.drop(['title'],axis=1,inplace=True)
df.drop(['ratings_disabled'],axis=1,inplace=True)
df.dtypes
#label encoding channel title
df['channel_title'].unique() 
label_encoder = preprocessing.LabelEncoder()
df['channel_title']= label_encoder.fit_transform(df['channel_title']) 
df['channel_title'].unique()




dcategory={1:"FILM AND ANIMATION",2:"AUTOS AND VEHICLE",10:"MUSIC",15:"PETS & ANIMALS",17:"SPORTS",18:"SHORT MOVIES",19:"TRAVEL & EVENTS",
           20:"GAMING",21:"VIDEOBLOGGING",22:"PEOPLE & BLOGS",23:"COMEDY",24:"ENTERTAINMENT",
           25:"NEWS & POLITICS",26:"HOWTO & STYLE",27:"EDUCATION",28:"SCIENCE & TECHNOLOGY",
           29:"TITLE NONPROFITS & ACTIVISM",30:"MOVIES",31:"ANIME/ANIMATION",
           32:"ACTION/ADVENTURE",33:"CLASSICS",34:"COMEDY",35:"DOCUMENTARY",
           36:"DRAMA",37:"FAMILY",38:"FOREIGN",39:"HORROR",40:"SCI-FI/FANTASY",
           41:"THRILLER",42:"SPORTS",43:"SHOWS",44:"TRAILERS"}

df['category_id'].replace(dcategory,inplace=True)
df['category_id'].describe()
df.to_csv('usmodd.csv')
df=pd.read_csv('usmodd.csv',encoding='latin1')
df.isnull().sum().sum()
df.describe()
df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
df.dtypes

#median views
df.loc[:,"views"].median()

#categorization

def views_categorization(views):
    if views < 685619.0:
        return "LOW"
    elif views >= 685619.0:
        return "HIGH"

#likes cat
df['likes'].describe()
df.loc[:,"likes"].median()

def likes_categorization(likes):
    if likes <= 5589.5:
        return "VERY LOW"
    elif likes >5589.5 and likes <=1844.0:
        return "LOW"
    elif likes >1844.0 and likes <=55126.5:
        return "HIGH"
    elif likes >55126.5 and likes <=5.023450:
        return "VERY HIGH"
    
#DISLIKES CATEGORIZATION
df['dislikes'].describe()
df.loc[:,"dislikes"].median()

def dislikes_categorization(dislikes):
    if dislikes <= 203.0:
        return "VERY LOW"
    elif dislikes >203.0 and dislikes <=632.0:
        return "LOW"
    elif dislikes >632.0 and dislikes <=1925.0:
        return "HIGH"
    elif dislikes >1925.0 and dislikes <=1643059:
        return "VERY HIGH"
    

df['views']=df['views'].apply(views_categorization)
df['views'].describe()
df['likes']=df['likes'].apply(likes_categorization)
df['dislikes']=df['dislikes'].apply(dislikes_categorization)


#one hot encoding
dict_views = {"HIGH" : 1, "LOW" : 0}
df['views'].replace(dict_views,inplace=True)
df.head()
final_df=pd.get_dummies(df, columns=['likes','dislikes'])

#label encoding category id
df['category_id'].unique() 
df['category_id']= label_encoder.fit_transform(df['category_id']) 
df['category_id'].unique()


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:43:16 2019

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
from sklearn.model_selection import train_test_split

import difflib as dff
import sklearn.metrics as metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import sklearn.linear_model as linear_model

#from sklearn.cross_validation import train_test_split

df = pd.read_csv('USMOD.csv',encoding='latin1')
df.head()
#df_summary=df.describe(include="all")
#df_summary
#df_summary.drop(['video_error_or_removed'], axis=1, inplace=True)
df.isnull().sum().sum()
df.dropna(how='any',inplace=True)
#df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)
#df_summary.drop(['thumbnail_link'], axis=1, inplace=True)


df.dtypes
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
df.drop(['channel_title'],axis=1,inplace=True)

df.dtypes

label_encoder = preprocessing.LabelEncoder()


df.isnull().sum().sum()
df.describe()
df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)

df.dtypes

df.to_csv('usap.csv')
df=pd.read_csv('usap.csv',encoding='latin1')
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
    elif likes >5589.5 and likes <= 18244.0:
        return "LOW"
    elif likes >18244.0 and likes <=  55126.5:
        return "HIGH"
    elif likes >55126.5 and likes <= 5023450:
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
df['likes']=df['likes'].apply(likes_categorization)
df['dislikes']=df['dislikes'].apply(dislikes_categorization)

#one hot encoding
dict_views = {"HIGH" : 1, "LOW" : 0}
df['views'].replace(dict_views,inplace=True)


final_df=pd.get_dummies(df, columns=['likes','dislikes'])
final_df.head()
final_df.dtypes
final_df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)

final_df.to_csv('encodeusa.csv')
feature_col=['category_id','days_to_trending','likes_HIGH','likes_LOW','likes_VERY HIGH','likes_VERY LOW','dislikes_HIGH','dislikes_LOW','dislikes_VERY HIGH','dislikes_VERY LOW']
X=final_df[feature_col]
y=final_df.views
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.head()
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

#consfusion matrix
confmatrix=metrics.confusion_matrix(y_test,y_pred)
confmatrix

print("accuracy:", metrics.accuracy_score(y_test,y_pred))
print("precision:", metrics.precision_score(y_test,y_pred))
print("recall:", metrics.recall_score(y_test,y_pred))


#y_pred_proba=logreg.predict_proba(X_test)[::,1]

from sklearn.metrics import roc_curve,auc
probs = logreg.predict_proba(X_test)
preds=probs [:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test,preds)
roc_auc = metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='AUC = %0.2f'%roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()
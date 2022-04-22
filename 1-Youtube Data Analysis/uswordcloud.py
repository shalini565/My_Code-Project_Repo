# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:54:14 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#%matplotlib inline

from wordcloud import WordCloud, STOPWORDS
df = pd.read_csv('usamap.csv',encoding='latin1') # updated to match uploaded set
df.head()
t1 = df["title"]

wordcloud_q = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=100,
                          max_font_size=40, 
                          random_state=1705
                         ).generate(str(t1))

def cloud_plot(wordcloud):
    fig = plt.figure(1, figsize=(20,15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
cloud_plot(wordcloud_q)

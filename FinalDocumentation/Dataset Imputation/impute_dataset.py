#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('FeatureCSV/traffic_sign_concat.csv', header=None)


# In[3]:


df.head(10)


# In[4]:


df_clean = df.replace([np.inf, -np.inf, 0], np.nan)

df_clean.isnull().sum()


# In[5]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
imputer.fit(df_clean)
df_imputed = imputer.transform(df_clean)


# In[6]:


df_imputed = pd.DataFrame(df_imputed)
df_imputed.isnull().sum()


# In[7]:


df_imputed.to_csv('FeatureCSV/traffic_sign_concat_imputed.csv', header=False, index=False)


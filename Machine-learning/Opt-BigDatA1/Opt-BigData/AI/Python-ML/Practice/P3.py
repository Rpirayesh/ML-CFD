# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:56:59 2019

@author: rpira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# We have imported all dependencied
df = pd.read_csv('data2.csv') # read data set using pandas

print(df.info()) # Overview of dataset



"""
df = df.drop(['Date'],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.

df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature
df_train = df[:1059]    # 60% training data and 40% testing data
df_test = df[1059:]

scaler = MinMaxScaler() # For normalizing dataset
print(scaler)

# We want to predict Close value of stock 
#X_train = scaler.fit_transform(df_train.drop(['Close'],axis=1).as_matrix())

y_train = scaler.fit_transform(df_train['Close'].as_matrix())



# y is output and x is features.
X_test = scaler.fit_transform(df_test.drop(['Close'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['Close'].as_matrix())
def denormalize(df,norm_data):
    df = df['Close'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    """
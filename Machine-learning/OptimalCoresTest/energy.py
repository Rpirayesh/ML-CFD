# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import time
start_time = time.time()

##Load the dataset
dataset= pd.read_csv('dataset.csv') # read data set using pandas
InputData=dataset.copy()

#Split features from labels
InputData.pop('P')
InputData.pop('D')
InputData.pop('Time')
InputData.pop('Error')
OutputData=InputData.pop('Energy')

train_dataset = InputData.sample(frac=.80,random_state=0)
test_dataset = InputData.drop(train_dataset.index)

train_labels = OutputData.sample(frac=.80,random_state=0)
test_labels = OutputData.drop(train_dataset.index)

#look at the overall statistics
Data_stats=dataset.describe()
Data_stats = Data_stats.transpose()
train_stats = train_dataset.describe()
Output_stats = test_labels.describe()
train_stats = train_stats.transpose()
Output_stats = Output_stats.transpose()

print(train_stats)
print(Output_stats)
print(Data_stats)

## Normalization Training
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


N0=2560
k0=layers.Dense(N0, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())])
N1=160
k1=layers.Dense(N1, activation=tf.nn.relu)
N2=160
k2=layers.Dense(N2, activation=tf.nn.relu)
kend=layers.Dense(1)
k=[k1]+[k2]
m=[k0]+k+[kend]
                        ## build the model
def build_model():
    model = keras.Sequential(m)
                        
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_absolute_percentage_error',
    optimizer=optimizer,
    metrics=['mean_squared_error','mean_absolute_percentage_error'])
    return model
model = build_model()

                        ## Using the the epoch approach  
EPOCHS = 100
                        
model = build_model()
                        
                        ## Using the cross validation approach
                        
                        # The patience parameter is the amount of epochs to check for improvement
                        
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100) 
#mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)                     
history =model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                                            validation_split = 0.2, verbose=1, callbacks=[early_stop])
                        ## saving the model
                        #MatrixModelAll.append(model)
                        ## Testing error
#saved_model = load_model('best_model.h5')
#loss, mse, mape = saved_model.evaluate(normed_test_data, test_labels, verbose=0)
#print(mape)

print("--- %s seconds ---" % (time.time() - start_time))

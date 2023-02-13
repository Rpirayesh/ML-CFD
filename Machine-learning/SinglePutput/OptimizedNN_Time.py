# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#raw_dataset = pd.read_csv('DataI300.csv') # read data set using pandas
#print(raw_dataset.info()) # Overview of dataset
#dataset = raw_dataset.copy()
#dataset.tail()
#dataset.isna().sum()
#dataset = dataset.dropna()
#print(dataset.tail())

##Split the data into train and test
#train_dataset = dataset.sample(frac=.99,random_state=0)
#test_dataset = dataset.drop(train_dataset.index)
#export_csv_Time = test_dataset.to_csv ("test_dataset.csv", index = None, header=True)

#export_csv_Time = test_dataset.to_csv ("test_dataset.csv", index = None, header=True) 
#export_csv_Time = train_dataset.to_csv ("train_dataset.csv", index = None, header=True)
#export_csv_Time = dataset.to_csv ("dataset.csv", index = None, header=True) 

##load data 
test_dataset= pd.read_csv('test_dataset.csv') # read data set using pandas
train_dataset= pd.read_csv('train_dataset.csv') # read data set using pandas
dataset= pd.read_csv('dataset.csv') # read data set using pandas


#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset.pop('Energy')
train_dataset.pop('Error')

test_dataset.pop('P')
test_dataset.pop('D')
test_dataset.pop('Energy')
test_dataset.pop('Error')

train_labels = train_dataset.pop('Time')
test_labels = test_dataset.pop('Time')

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

## build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(630, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),

#    layers.Dense(620, activation=tf.nn.relu),
    layers.Dense(520, activation=tf.nn.relu),
    layers.Dense(420, activation=tf.nn.relu),
    layers.Dense(310, activation=tf.nn.relu),
    layers.Dense(130, activation=tf.nn.relu),
    layers.Dense(110, activation=tf.nn.relu),
    layers.Dense(80, activation=tf.nn.relu),
    layers.Dense(20, activation=tf.nn.relu),      


    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_absolute_percentage_error',
                optimizer=optimizer,
                metrics=['mean_squared_error','mean_absolute_percentage_error'])
  return model
model = build_model()

## testing the model
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

## Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
  
## Using the the epoch approach  
EPOCHS = 700

model = build_model()

## Using the cross validation approach

# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=400)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

## Testing error
loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Energy".format(mse))

print(mse)
print(mape)
## Make prediction

test_predictions_Time = model.predict(normed_test_data).flatten()
error = test_predictions_Time - test_labels


## Variables to save

Variable=[mse,mape]

### Save files
np.savetxt("VariableOpt_Time.csv",Variable )
np.savetxt("test_predictions_Time.csv",test_predictions_Time )
export_csv_error = error.to_csv ("error_prediction_Time.csv", index = None, header=True) 


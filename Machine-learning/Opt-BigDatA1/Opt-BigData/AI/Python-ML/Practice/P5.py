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
##Reading the data set

raw_dataset = pd.read_csv('data2.csv') # read data set using pandas
print(raw_dataset.info()) # Overview of dataset
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
print(dataset.tail())

#Split the data into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Inspect the data
sns.pairplot(train_dataset[["Eu1", "Eu2", "Eu3","AngD1","AngD2","AngD3", "Energy","Error","P","D","Time"]], diag_kind="kde")
plt.show()

#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset.pop('Time')
test_dataset.pop('P')
test_dataset.pop('D')
test_dataset.pop('Time')
train_labels = train_dataset.pop('Energy')
test_labels = test_dataset.pop('Energy')
#look at the overall statistics
train_stats = train_dataset.describe()
Output_stats = test_labels.describe()
train_stats = train_stats.transpose()
Output_stats = Output_stats.transpose()

print(train_stats)
print(Output_stats)

## Normalization
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

## build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_absolute_percentage_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error','mean_absolute_percentage_error'])
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
    

## Plottig the metrics
def plot_history(history):
  hist = pd.DataFrame(history.history)
  print(hist)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Energy]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('mean_absolute_percentage_error [$Energy^2$]')
  plt.plot(hist['epoch'], hist['mean_absolute_percentage_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_percentage_error'],
           label = 'Val Error')
  plt.ylim([0,12000])
  plt.legend()
  plt.show()   
  
## Using the the epoch approach  
EPOCHS = 100

model = build_model()

## Using the cross validation approach

# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
## Testing error
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Energy".format(mae))

## Make prediction

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Energy]')
plt.ylabel('Predictions [Energy]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

## error distribution
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Energy]")
_ = plt.ylabel("Count")
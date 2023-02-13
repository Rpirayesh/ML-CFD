# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:29:06 2019

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

## Keras part
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

## GridSearchCV

from keras.activations import relu, sigmoid
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.utils import np_utils

##Reading the data set
#maxP=120000
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
Data_stats=dataset.describe()
Data_stats = Data_stats.transpose()
train_stats = train_dataset.describe()

## Normalization Training
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Function with hyperparameters to optimize
def create_model(optimizer='adam', activation = 'sigmoid', hidden_layers=2):
  # Initialize the constructor
    model = Sequential()
      # Add an input layer
    model.add(Dense(32, activation=activation, input_shape=784))

    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(Dense(16, activation=activation))

      # Add an output layer 
    model.add(Dense(num_classes, activation='softmax'))
      #compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=
     ['accuracy'])
    return model

activations = [sigmoid, relu]
param_grid = dict(hidden_layers=3,activation=activations, batch_size = [256], epochs=[30])
grid = GridSearchCV(estimator=modelCV, param_grid=param_grid, scoring='accuracy')
grid_result = grid.fit(X_train, Y_train)
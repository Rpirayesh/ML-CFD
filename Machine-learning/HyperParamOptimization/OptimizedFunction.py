# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import time
t0= time.clock()
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping


#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import EarlyStopping

# load a saved model
from tensorflow.keras.models import load_model

#saved_model = load_model('best_model.h5')

##load data 
test_dataset= pd.read_csv('test_dataset.csv') # read data set using pandas
train_dataset= pd.read_csv('train_dataset.csv') # read data set using pandas
dataset= pd.read_csv('dataset.csv') # read data set using pandas
#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset_Time=train_dataset.pop('Time')
train_dataset.pop('Error')
test_dataset.pop('P')
test_dataset.pop('D')
test_dataset_Time=test_dataset.pop('Time')
test_dataset.pop('Error')
train_labels = train_dataset.pop('Energy')
test_labels = test_dataset.pop('Energy')

#look at the overall statistics
Data_stats=dataset.describe()
Data_stats = Data_stats.transpose()
train_stats = train_dataset.describe()
Output_stats = test_labels.describe()
train_stats = train_stats.transpose()
Output_stats = Output_stats.transpose()

#print(train_stats)
#print(Output_stats)
#print(Data_stats)

## Normalization Training
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

## build the model

def create_model(ModelInfo):
    ### Defining reguralization technique
    
#    reg = l1(0.001)
    model = keras.Sequential()
    model.add(layers.Dense(ModelInfo['Nerouns'][0], input_shape=[len(train_dataset.keys())],
                           activation=ModelInfo['Activation_Method'][0],
                           kernel_initializer =ModelInfo['W_Initialization_Method'][0],
                            bias_initializer = ModelInfo['W_Initialization_Method'][0],
                            activity_regularizer=ModelInfo['Reguralization'][0],
    kernel_constraint=ModelInfo['kernel_constraint'][0])),
    model.add(layers.Dropout(ModelInfo['Dropout_Value'][0])),
    for c in range(1,ModelInfo['Layers'][0]):
#        
        model.add(layers.Dense(ModelInfo['Nerouns'][c],
                               activation=tf.nn.relu, 
                               kernel_initializer =ModelInfo['W_Initialization_Method'][0],
                            bias_initializer = ModelInfo['W_Initialization_Method'][0],
                            activity_regularizer=ModelInfo['Reguralization'][0],
                            kernel_constraint=ModelInfo['kernel_constraint'][0])),
        model.add(layers.Dropout(ModelInfo['Dropout_Value'][0])),
        model.add(layers.BatchNormalization()),
        
    
    model.add(layers.Dense(1, activation=ModelInfo['Activation_Method'][0])),
    return model


def compile_model(ModelInfo):

####Call the model and assign the loss function to it
    model=create_model(ModelInfo)

    model.compile(loss='mean_absolute_percentage_error',
                                        optimizer=ModelInfo['optimizer'][0],
                                        metrics=['mean_squared_error','mean_absolute_percentage_error'])
## Using the cross validation approach

###########Callbacks
# The patience parameter is the amount of epochs to check for improvement

    early_stop =EarlyStopping(monitor='val_loss', patience=2)
    #mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
#Train the model and save it
    model.fit(normed_train_data, train_labels,batch_size=ModelInfo['Batches'][0], epochs=ModelInfo['Epochs'][0],
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, reduce_lr])
    #saved_model = load_model('best_model.h5')
    loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
    return model, mape 


### Make the dictionary for the values
#ModelInfo={}
#ModelInfo['Nerouns']=[200,80]
#ModelInfo['Layers']=[2,3]
#ModelInfo['Dropout_Value']=[0,0.2,0,4,0.8]
#ModelInfo['Reguralization']=[l1(0.1),l2(0.2)]
#ModelInfo['kernel_constraint']=[max_norm(3),max_norm(5)]
#ModelInfo['Activation_Method']=['relu','sigmoid']
#ModelInfo['Epochs']=[700,1200]
#ModelInfo['Batches']=[32,64]
#ModelInfo['optimizer']=['Nadam','Adam']
#ModelInfo['W_Initialization_Method']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]

#filehandler = open('filename_pi.obj', 'rb') 
#ModelInfo = pickle.load(filehandler)
#
#saved_model, model =compile_model(ModelInfo)
#
### Testing error
#loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
#loss2, mse2, mape2 = saved_model.evaluate(normed_test_data, test_labels, verbose=0)
#print("Testing set Mean Abs Error: {:5.2f} Energy".format(mse))
#t1 = time.clock() - t0
#print(t1)
### Make prediction
#
#test_predictions = model.predict(normed_test_data).flatten()
## Loading the model
#saved_model = load_model('best_model.h5')

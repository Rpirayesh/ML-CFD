# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""
from __future__ import absolute_import, division, print_function, unicode_literals
#group rank and group size and type of output
#import time
#t0= time.clock()
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
import math
# load a saved model
from tensorflow.keras.models import load_model
################Making the database
import pickle

#saved_model = load_model('best_model.h5')

##load data 
dataset= pd.read_csv('DataI300Q7893.csv') # read data set using pandas


#np.random.shuffle(dataset.values)
#dataset.sample(frac=1)

def norm(x,stats):
  return (x - stats['mean']) / stats['std']

def Output_moddel_Data(Portion, Feature):
    ## Deviding the data into the portion
    ppData=len(dataset)
    PortionData=math.floor(ppData*Portion)
    DataPortion=dataset[0:int(PortionData)]
    ## Making the input
    InputData=DataPortion.copy()
    InputData.pop('P')
    InputData.pop('D')
    InputData.pop('Time')
    InputData.pop('Error')
    InputData.pop('Energy')
    ## Making the output
    Out=DataPortion.copy()
    Output=Out[Feature]
    ##Making the Parameter model
#    Model=ParamDB[Feature]
    File=Feature+'Param.obj'
    filehandler = open(File, 'rb') 
    Model=pickle.load(filehandler)

    return InputData,Output,Model

#InputData,Output,Model=Output_moddel_Data(Portion, Feature)

def CR(CrossCount,K_fold,InputData,Output):
    pp=len(InputData)

    counter=math.floor(pp/K_fold)    
    ##### Making the data
    
    IndEnd=(CrossCount+1)*counter
    IndBeging=CrossCount*counter
#    print(IndEnd)
    test_dataset = InputData[int(IndBeging):int(IndEnd)]
    train_dataset = InputData.drop(test_dataset.index)
    
    
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    
#    test_stats = test_dataset.describe()
#    test_stats = train_stats.transpose()
    
    test_labels = Output[int(IndBeging):int(IndEnd)]
    train_labels = Output.drop(test_labels.index)
    ### Calling the normalized data
    normed_train_data = norm(train_dataset,train_stats)
    normed_test_data = norm(test_dataset,train_stats) 
    
    return normed_train_data,normed_test_data,train_labels,test_labels
    
## build the model

def create_model(ModelInfo):
    ### Defining reguralization technique
    
#    reg = l1(0.001)
    model = keras.Sequential()
    model.add(layers.Dense(ModelInfo['Nerouns'][0], input_shape=[7],
                           activation=ModelInfo['Activation_Method'][0],
#                           kernel_initializer =ModelInfo['W_Initialization_Method'][0],
#                            bias_initializer = ModelInfo['W_Initialization_Method'][0],
#                            activity_regularizer=ModelInfo['Reguralization'][0],
                             activity_regularizer=l1(ModelInfo['Reguralization'][0]),

#    kernel_constraint=ModelInfo['kernel_constraint'][0])),
     kernel_constraint=max_norm(ModelInfo['kernel_constraint'][0]))),
    model.add(layers.Dropout(ModelInfo['Dropout_Value'][0])),
    for c in range(1,ModelInfo['Layers'][0]):
        print('Index=',c) 
        model.add(layers.Dense(ModelInfo['Nerouns'][c],
                               activation=tf.nn.relu, 
#                               kernel_initializer =ModelInfo['W_Initialization_Method'][0],
#                            bias_initializer = ModelInfo['W_Initialization_Method'][0],
#                            activity_regularizer=ModelInfo['Reguralization'][c],
                             activity_regularizer=l1(ModelInfo['Reguralization'][c]),
#                            kernel_constraint=ModelInfo['kernel_constraint'][c])),
                             kernel_constraint=max_norm(ModelInfo['kernel_constraint'][c]))),
        model.add(layers.Dropout(ModelInfo['Dropout_Value'][c])),
#        model.add(layers.BatchNormalization()),
        
    
    model.add(layers.Dense(1, activation=ModelInfo['Activation_Method'][0])),
    return model


def compile_model(CountParam,CrossCount,K_fold,InputData,Output,Model):
    m=len(Model[0])
    Indx=int(np.floor(CountParam/m))
    Indy=CountParam-Indx*m-1
#    ModelInfo=Model[int(Indx)][int(Indy)]
    ModelInfo=Model[CountParam]
    
    print('ModelInfo=',ModelInfo)
####Call the model and assign the loss function to it
    print("\n***************Creating the model******************")
    model=create_model(ModelInfo)

    print("\n***************Compiling the model******************")
    model.compile(loss='mean_absolute_percentage_error',
                                        optimizer=ModelInfo['optimizer'][0],
                                        metrics=['mean_squared_error','mean_absolute_percentage_error'])
## Using the cross validation approach

###########Callbacks
# The patience parameter is the amount of epochs to check for improvement

    early_stop =EarlyStopping(monitor='val_loss', patience=50)
#    mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    
## Provide the data within the cross fold
    print("\n***************CR function******************")
    normed_train_data,normed_test_data,train_labels,test_labels=CR(CrossCount,K_fold,InputData,Output)
    
    #Train the model and save it
    print("\n***************Fitting the model******************")
    print(normed_train_data.head())
    model.fit(normed_train_data, train_labels,batch_size=ModelInfo['Batches'][0], epochs=ModelInfo['Epochs'][0],
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, reduce_lr])

    print("\n***************Saving the model******************")
#    saved_model = load_model('best_model.h5')
#Test the model
    loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
    return ModelInfo, model,  mape 

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
steps=4
MapeBench=100 
##load data 
test_dataset= pd.read_csv('test_dataset.csv') # read data set using pandas
train_dataset= pd.read_csv('train_dataset.csv') # read data set using pandas
dataset= pd.read_csv('dataset.csv') # read data set using pandas


#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset_Time=train_dataset.pop('Time')
train_dataset.pop('Energy')
test_dataset.pop('P')
test_dataset.pop('D')
test_dataset_Time=test_dataset.pop('Time')
test_dataset.pop('Energy')
train_labels = train_dataset.pop('Error')
test_labels = test_dataset.pop('Error')

#look at the overall statistics
Data_stats=dataset.describe()
Data_stats = Data_stats.transpose()
train_stats = train_dataset.describe()
Output_stats = test_labels.describe()
train_stats = train_stats.transpose()
Output_stats = Output_stats.transpose()

## Normalization Training
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

MatrixAll=[] ###Matrix for definign the minimum of minimization
MatrixModel=[] #### Matrix for the models
MatrixModelIteration=[]
for i0 in range(2):
    for i1 in range(5):
        for i2 in range(0,i1+1):
            for i3 in range(0,i2+1):
                for i4 in range(0,i3+1):
                    for i5 in range(1,4):
        ##### Make the layers as variables
        #### make the complete layers to choose layers from
                        N0=10*steps**i1
                        k0=layers.Dense(N0, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())])
                        N1=10*steps**i2
                        k1=layers.Dense(N1, activation=tf.nn.relu)
                        N2=10*steps**i3
                        k2=layers.Dense(N2, activation=tf.nn.relu)
                        N3=10*steps**i4
                        k3=layers.Dense(N3, activation=tf.nn.relu)
                        kend=layers.Dense(1)
                        k=[k1]+[k2]+[k3]
#			k=[k1]+[k2]
                        LayerVariable=i5
                        m=[k0]+k[0:LayerVariable]+[kend]
                        ## build the model
                        def build_model():
                          model = keras.Sequential(m)
                        
                          optimizer = tf.keras.optimizers.RMSprop(0.001)
                        
                          model.compile(loss='mean_absolute_percentage_error',
                                        optimizer=optimizer,
                                        metrics=['mean_squared_error','mean_absolute_percentage_error'])
                          return model
                        model = build_model()
                        
                        ## Train the model
                        # Display training progress by printing a single dot for each completed epoch
                        class PrintDot(keras.callbacks.Callback):
                          def on_epoch_end(self, epoch, logs):
                            if epoch % 100 == 0: print('')
                            print('.', end='')
                            
                          
                        ## Using the the epoch approach  
                        EPOCHS = 1200
                        
                        model = build_model()
                        
                        ## Using the cross validation approach
                        
                        # The patience parameter is the amount of epochs to check for improvement
                        
                        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=240)                      
                        model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
                        ## saving the model
                        #MatrixModelAll.append(model)
                        ## Testing error
                        loss, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
                        
                        ## Making the list and matrix saving the optimal MAPE and MSE
                        MatrixLayersMseMape=[i1,i2,i3,i4,i5,mse,mape]
                        MatrixAll.append(MatrixLayersMseMape)
                        ## Compare the MapeBench and find the 
                        if mape<MapeBench:

                            MapeBench=mape
                            MinMatrix=MatrixLayersMseMape
                            print(MinMatrix)
			    MinModel=model
    MatrixModelIteration.append(MinMatrix)
                        
test_predictions = MinModel.predict(normed_test_data).flatten()
error = test_predictions - test_labels
## Save files
np.savetxt("test_predictions_Error.csv",test_predictions )
export_csv_error = error.to_csv ("error_prediction_Error.csv", index = None, header=True) 

np.savetxt("MatrixAll_Error.csv",MatrixAll ) 
np.savetxt("MatrixMin_Error.csv",MinMatrix )
np.savetxt("MatrixMinIter_Error.csv",MatrixModelIteration )
MinModel.save("ModelError.model")


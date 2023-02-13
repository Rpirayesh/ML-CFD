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
MapeBench=100 
steps=4
#raw_dataset = pd.read_csv('DataI300.csv') # read data set using pandas
#print(raw_dataset.info()) # Overview of dataset
#dataset = raw_dataset.copy()
#dataset.tail()
#dataset.isna().sum()
#dataset = dataset.dropna()
#print(dataset.tail())

##Split the data into train and test
#train_dataset = dataset.sample(frac=.8,random_state=0)
#test_dataset = dataset.drop(train_dataset.index)
#export_csv_Time = test_dataset.to_csv ("test_dataset.csv", index = None, header=True)

#export_csv_Time = test_dataset.to_csv ("test_dataset.csv", index = None, header=True) 
#export_csv_Time = train_dataset.to_csv ("train_dataset.csv", index = None, header=True)
#export_csv_Time = dataset.to_csv ("dataset.csv", index = None, header=True) 

##load data 
test_dataset= pd.read_csv('test_dataset.csv') # read data set using pandas
train_dataset= pd.read_csv('train_dataset.csv') # read data set using pandas
dataset= pd.read_csv('dataset.csv') # read data set using pandas
train_labels=train_dataset.copy()
test_labels=test_dataset.copy()

#Split features from labels
train_dataset.pop('P')
train_dataset.pop('D')
train_dataset_Time=train_dataset.pop('Time')
test_dataset.pop('P')
test_dataset.pop('D')
test_dataset_Time=test_dataset.pop('Time')
train_dataset.pop('Error')
test_dataset.pop('Error')
train_dataset.pop('Energy')
test_dataset.pop('Energy')
#### Labels
train_labels.pop('P')
train_labels.pop('D')
train_labels.pop('Prop')
train_labels.pop('Eu1')
train_labels.pop('Eu2')
train_labels.pop('Eu3')
train_labels.pop('AngD1')
train_labels.pop('AngD2')
train_labels.pop('AngD3')
#train_labels.pop('Time')
#train_labels.pop('Error')
#train_labels.pop('Energy')

test_labels.pop('P')
test_labels.pop('D')
test_labels.pop('Prop')
test_labels.pop('Eu1')
test_labels.pop('Eu2')
test_labels.pop('Eu3')
test_labels.pop('AngD1')
test_labels.pop('AngD2')
test_labels.pop('AngD3')
#test_labels.pop('Time')
#test_labels.pop('Error')
#test_labels.pop('Energy')

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

optimizer = tf.keras.optimizers.SGD(0.001)

MatrixAll=[] ###Matrix for definign the minimum of minimization
MatrixModel=[] #### Matrix for the models
MatrixModelIteration=[]
for i0 in range(1):
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
                        kend=layers.Dense(3)
                        k=[k1]+[k2]+[k3]
#			k=[k1]+[k2]
                        LayerVariable=i5
                        m=[k0]+k[0:LayerVariable]+[kend]
                        ## build the model
                        def build_model():
                          model = keras.Sequential(m)
                        
                          
                        
                          model.compile(loss='mean_squared_error',
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
                        
                        #print(mape)
                        ## Making the list and matrix saving the optimal MAPE and MSE
                        MatrixLayersMseMape=[i1,i2,i3,i4,i5,mse,mape]
                        MatrixAll.append(MatrixLayersMseMape)
#                        MatrixAllNP=np.matrix(MatrixAll)
                        ## Compare the MapeBench and find the 
                        if mape<MapeBench:
                            MapeBench=mape
                            MinMatrix=MatrixLayersMseMape
			    MinModel=model
                            print(MinMatrix)
    MatrixModelIteration.append(MinMatrix)
                        
test_predictions = model.predict(normed_test_data)
error = test_predictions - test_labels
## 
#export_csv_Time = error.to_csv ("Multi_ErrorAndEnergy_error.csv", index = None, header=True)

## Save files
#np.savetxt("VariableOpt_Energy.csv",Variable )

np.savetxt("test_predictions_EnergyAndErrorTime.csv",test_predictions )
export_csv_error = error.to_csv ("EnergyAndErrorTime_error.csv", index = None, header=True) 

np.savetxt("MatrixAll_EnergyAndErrorTime.csv",MatrixAll ) 
np.savetxt("MatrixMin_EnergyAndErrorTime.csv",MinMatrix )
np.savetxt("MatrixMinIter_EnergyAndErrorTime.csv",MatrixModelIteration )
MinModel.save("ModelEnergyAndErrorTime.model")



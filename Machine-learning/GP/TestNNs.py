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
from tensorflow.keras.models import load_model  # to load a model
import numpy as np
from operator import truediv
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
#train_labels.pop('P')
#train_labels.pop('D')
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

#test_labels.pop('P')
#test_labels.pop('D')
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

### Defining MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(test_labels)

############################### Making the power from data
### Making the true power
y_true_power_notScaled=test_labels['Energy'].divide(test_labels['Time'])
y_true=[1/3600 * i for i in y_true_power_notScaled]

### Loading models

## Energy Error Time P D
#Indices
IndP=0
IndD=IndP+1
IndTime=IndD+1
IndEnergy=IndTime+1
IndError=IndEnergy+1
#
print('Energy Error Time P D')
print('')
MinModelEnergyErrorTimePD=load_model('ModelEnergyErrorTimePD.h5')

loss, mse, mape = MinModelEnergyErrorTimePD.evaluate(normed_test_data, test_labels, verbose=0)
print('MinModelEnergyErrorTimePD_HyperParam=')
ModelHyper=MinModelEnergyErrorTimePD.summary()
print('MinModelEnergyErrorTimePD=',mape)

######## Mape for each output
print('')
test_predictions = MinModelEnergyErrorTimePD.predict(normed_test_data)
EnergyENErTiPDPredicted=mean_absolute_percentage_error(test_labels['Energy'],test_predictions[:,IndEnergy])
print('Energy_Estimation_Time-Energy-Erro-P-D=',EnergyENErTiPDPredicted)
TimeENErTiPDPredicted=mean_absolute_percentage_error(test_labels['Time'],test_predictions[:,IndTime])
print('TimeEstimation_Time-Energy-Erro-P-D=',TimeENErTiPDPredicted)
ErrorENErTiPDPredicted=mean_absolute_percentage_error(test_labels['Error'],test_predictions[:,IndError])
print('ErrorEstimation_Time-Energy-Erro-P-D=',ErrorENErTiPDPredicted)


PENErTiDPredicted=mean_absolute_percentage_error(test_labels['P'],test_predictions[:,IndP])
print('P_Estimation_Time-Energy-Erro-P-D=',PENErTiDPredicted)
DENErTiPPredicted=mean_absolute_percentage_error(test_labels['D'],test_predictions[:,IndD])
print('D_Estimation_Time-Energy-Erro-P-D=',DENErTiPPredicted)


######## Mape for power

y_pred_power_notScaled=list(map(truediv,test_predictions[:,IndEnergy],test_predictions[:,IndTime]))
y_pred=[1/3600 * i for i in y_pred_power_notScaled]
PowerEstimated_EnergyTimeErrorPD=mean_absolute_percentage_error(y_true, y_pred)
print('PowerEstimated_EnergyTimeError=',PowerEstimated_EnergyTimeErrorPD)



### Loading models

## Energy Error Time
# Indices
#IndP=0
#IndD=IndP+1
IndTime=0
IndEnergy=IndTime+1
IndError=IndEnergy+1
#
print('Energy Error Time')
print('')
MinModelEnergyErrorTime=load_model('ModelEnergyAndErrorTime.h5')

loss, mse, mape = MinModelEnergyErrorTime.evaluate(normed_test_data, test_labels[['Time','Energy','Error']], verbose=0)
print('MinModelEnergyErrorTime_HyperParam=')
ModelHyper=MinModelEnergyErrorTime.summary()
print('MinModelEnergyErrorTime=',mape)

######## Mape for each output
print('')
test_predictions = MinModelEnergyErrorTime.predict(normed_test_data)
EnergyENErTiPredicted=mean_absolute_percentage_error(test_labels['Energy'],test_predictions[:,IndEnergy])
print('Energy_Estimation_Time-Energy-Erro=',EnergyENErTiPredicted)
TimeENErTiPredicted=mean_absolute_percentage_error(test_labels['Time'],test_predictions[:,IndTime])
print('TimeEstimation_Time-Energy-Erro=',TimeENErTiPredicted)
ErrorENErTiPredicted=mean_absolute_percentage_error(test_labels['Error'],test_predictions[:,IndError])
print('ErrorEstimation_Time-Energy-Erro=',ErrorENErTiPredicted)

######## Mape for power

y_pred_power_notScaled=list(map(truediv,test_predictions[:,1],test_predictions[:,0]))
y_pred=[1/3600 * i for i in y_pred_power_notScaled]
PowerEstimated_EnergyTimeError=mean_absolute_percentage_error(y_true, y_pred)
print('PowerEstimated_EnergyTimeError=',PowerEstimated_EnergyTimeError)

# Energy Time
# Indices
#IndP=0
#IndD=IndP+1
IndTime=0
IndEnergy=IndTime+1
#IndError=IndEnergy+1
#

print('')
print('Energy Time')
MinModelEnergyTime=load_model('ModelEnergyAndTime2.h5')
loss, mse, mape = MinModelEnergyTime.evaluate(normed_test_data, test_labels[['Time','Energy']], verbose=0)
print('MinModelEnergyErrorTime_HyperParam=')
ModelHyper=MinModelEnergyTime.summary()
print('MinModelEnergyTime=',mape)
########### MAPE of each output
test_predictions = MinModelEnergyTime.predict(normed_test_data)
EnergyENTiPredicted=mean_absolute_percentage_error(test_labels['Energy'],test_predictions[:,1])
print('EnergyEstimation=',EnergyENTiPredicted)
EnergyENErTiPredicted=mean_absolute_percentage_error(test_labels['Time'],test_predictions[:,0])
print('TimeEstimation=',EnergyENErTiPredicted)

###### Power
y_pred_power_notScaled=list(map(truediv,test_predictions[:,1],test_predictions[:,0]))
y_pred=[1/3600 * i for i in y_pred_power_notScaled]
PowerEstimated_EnergyTime=mean_absolute_percentage_error(y_true, y_pred)
print(PowerEstimated_EnergyTime)

## Energy Error
# Indices
#IndP=0
#IndD=IndP+1
#IndTime=0
IndEnergy=0
IndError=IndEnergy+1
#

print('')
print('Energy Error')
MinModelEnergError=load_model('ModelErrorAndEnergy.h5')
loss, mse, mape = MinModelEnergError.evaluate(normed_test_data, test_labels[['Energy','Error']], verbose=0)
print('MinModelEnergError_HyperParam=')
ModelHyper=MinModelEnergError.summary()
print('MinModelEnergError=',mape)

########### MAPE of each output
test_predictions = MinModelEnergError.predict(normed_test_data)
EnergyENTiPredicted=mean_absolute_percentage_error(test_labels['Energy'],test_predictions[:,0])
print('EnergyEstimation=',EnergyENTiPredicted)
EnergyENErTiPredicted=mean_absolute_percentage_error(test_labels['Error'],test_predictions[:,1])
print('ErrorEstimation=',EnergyENErTiPredicted)

## Energy 
# Indices
#IndP=0
#IndD=IndP+1
#IndTime=0
IndEnergy=0
#IndError=IndEnergy+1
#

print('')
print('Energy ')
MinModelEnergy=load_model('ModelEnergy.model')
loss, mse, mape = MinModelEnergy.evaluate(normed_test_data, test_labels['Energy'], verbose=0)
print('MinModelEnergy=')
ModelHyper=MinModelEnergy.summary()
print('MinModelEnergy=',mape)

## Time
# Indices
#IndP=0
#IndD=IndP+1
IndTime=0
#IndEnergy=IndTime+1
#IndError=IndEnergy+1
#

print('')
print('Time')
MinModelTime=load_model('ModelTime.model')
loss, mse, mape = MinModelTime.evaluate(normed_test_data, test_labels['Time'], verbose=0)
print('MinModelTime=')
ModelHyper=MinModelTime.summary()
print('MinModelTime=',mape)

###### Power
test_predictions_Energy = MinModelEnergy.predict(normed_test_data)
test_predictions_Time = MinModelTime.predict(normed_test_data)
y_pred_EnergyTime=list(map(truediv,test_predictions_Energy,test_predictions_Time))
y_pred_1output=y_pred=[1/3600 * i for i in y_pred_EnergyTime]
PowerEstimated_EnergyTime_1output=mean_absolute_percentage_error(y_true, y_pred_1output)
print(PowerEstimated_EnergyTime_1output)

## Error
# Indices
#IndP=0
#IndD=IndP+1
#IndTime=0
#IndEnergy=IndTime+1
IndError=0
#

print('')
print('Error')
MinModelError=load_model('ModelError.model')
loss, mse, mape = MinModelError.evaluate(normed_test_data, test_labels['Time'], verbose=0)
print('MinModelError=')
ModelHyper=MinModelError.summary()
print('MinModelError=',mape)

## P
# Indices
IndP=0
#IndD=IndP+1
#IndTime=0
#IndEnergy=IndTime+1
#IndError=IndEnergy+1
#

print('')
print('P')
MinModelP=load_model('ModelP.model')
loss, mse, mape = MinModelError.evaluate(normed_test_data, test_labels['P'], verbose=0)
print('MinModelP=')
ModelHyper=MinModelP.summary()
print('MinModelP=',mape)

## D
# Indices
#IndP=0
IndD=0
#IndTime=0
#IndEnergy=IndTime+1
#IndError=IndEnergy+1
#

print('')
print('D')
MinModelD=load_model('ModelD.model')
loss, mse, mape = MinModelError.evaluate(normed_test_data, test_labels['D'], verbose=0)
print('MinModelD=')
ModelHyper=MinModelD.summary()
print('MinModelD=',mape)


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:08:18 2020

@author: rpira
"""


from __future__ import absolute_import, division, print_function, unicode_literals
#group rank and group size and type of output
import pandas as pd
import sklearn.gaussian_process as gp
import math
import numpy as np
####### data for crossfold
dataset= pd.read_csv('CFD_data.csv') # read data set using pandas
datasetTest=pd.read_csv('CFDAnalasis2.csv')  # read the test data set using pandas

#np.random.shuffle(dataset.values)
#dataset.sample(frac=1)
InputData=dataset.copy()
# InputData=InputData.dropna
#Split features from labels
# InputData.pop('P')
# InputData.pop('D')
# InputData.pop('Time')
# InputData.pop('Alpha')
OutputData=InputData.pop('Alpha')

#Define the portion for the data
Feature = 'Nu'
K_fold=3
Portion = 1
def Output_moddel_Data(CrossCount):
    # Deviding the data into the portion
    ppData = len(dataset)
    PortionData = math.floor(ppData*Portion)
    DataPortion = dataset[0:int(PortionData)]

    # Making the input

    InputData = DataPortion.copy()
    # InputData.pop('P')
    # InputData.pop('D')
    # InputData.pop('Time')
    # InputData.pop('Error')
    # InputData.pop('Energy')
    InputData.pop('Nu')
    
    # Making the output
    Out = DataPortion.copy()
    Output = Out[Feature]
    pp = len(InputData)
    counter = math.floor(pp/K_fold)
    # Making the data
    IndEnd = (CrossCount+1)*counter
    IndBeging = CrossCount*counter
    test_dataset = InputData[int(IndBeging):int(IndEnd)]
    train_dataset = InputData.drop(test_dataset.index)
    test_labels = Output[int(IndBeging):int(IndEnd)]
    train_labels = Output.drop(test_labels.index)
    return train_dataset, test_dataset, train_labels, test_labels

# pp=len(InputData)
# counter=math.floor(pp/K_fold)
# i=0
# IndEnd=(i+1)*counter
# IndBeging=i*counter
# test_dataset = InputData[int(IndBeging):int(IndEnd)]
# train_dataset = InputData.drop(test_dataset.index)
# test_labels = OutputData[int(IndBeging):int(IndEnd)]
# train_labels = OutputData.drop(test_labels.index)
### Defining MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

CrossCount=2
index=[0, 2, 0, 1]
train_dataset, test_dataset, train_labels, test_labels=Output_moddel_Data(CrossCount)
## Making the GP
kernel = gp.kernels.RBF(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
model.fit(train_dataset,train_labels)
params = model.kernel_.get_params()
y_pred, std = model.predict(test_dataset, return_std=True)
y_test=model.predict(datasetTest)
MSE = ((y_pred-test_labels)**2).mean()
MAPE=mean_absolute_percentage_error(test_labels, y_pred)
print("MAPE=",MAPE)
print("MSE=",MSE)
print("Predicttion=", y_test)
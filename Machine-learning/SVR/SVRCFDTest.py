# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:08:18 2020

@author: rpira
"""
import pandas as pd
from sklearn.svm import SVR
import math
import numpy as np
from random import choice
import csv
# data for crossfold, feature, portion,K_fold
dataset = pd.read_csv('CFD_data.csv')  # read data set using pandas
datasetTest=pd.read_csv('CFDAnalasis2.csv')  # read the test data set using pandas
Feature = 'Nu'
# print('Output=', Feature)
Portion = 1
K_fold = 3
# Defining the dictionary
Dict = {}
Dict['Nu'] = {"kernel":  ['rbf', 'linear', 'sigmoid'], "C": [1, 10, 100, 1000, 10000],
                  "gamma": [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],
                  "epsilon": [0.1, 1e-2, 1e-3, 1e-4]}

# Defining MAPE

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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Defining the model


def Model(train_dataset, train_labels, test_dataset, test_labels, datasetTest, index):
    print("Ind", index)
# The SVR Model
    regressor = SVR(kernel=Dict[Feature]['kernel'][index[0]],
                    C=Dict[Feature]['C'][index[1]],
                    gamma=Dict[Feature]['gamma'][index[2]],
                    epsilon=Dict[Feature]['epsilon'][index[3]])
# Fitting the lodel
    regressor.fit(train_dataset, train_labels)
# Prediction from the model
    y_pred = regressor.predict(test_dataset)
# Obtaining the MSE and MAPE
    MSE = ((y_pred-test_labels)**2).mean()
    MAPE = mean_absolute_percentage_error(test_labels, y_pred)
    y_test=regressor.predict(datasetTest)
    return MAPE, MSE, y_test

# Function for giving us the input and output based on the selected output
CrossCount=2
index=[0, 2, 0, 1]
train_dataset, test_dataset, train_labels, test_labels=Output_moddel_Data(CrossCount)

MAPE, MSE, y_test=Model(train_dataset, train_labels, test_dataset, test_labels, datasetTest, index)
print("")
print("mape=",MAPE)
print("")
print("MSE=",MSE)
print("Predicttion=", y_test)


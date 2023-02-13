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
Feature = 'Nu'
print('Output=', Feature)
Portion = 1
K_fold = 3
Metrcis = Feature+'MAPEmse.csv'
# Defining the dictionary
Dict = {}
Dict['Nu'] = {"kernel":  ['rbf', 'linear', 'sigmoid'], "C": [1, 10, 100, 1000, 10000],
                  "gamma": [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],
                  "epsilon": [0.1, 1e-2, 1e-3, 1e-4]}

# Defining MAPE


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Defining the model


def Model(train_dataset, train_labels, test_dataset, test_labels, index):
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
    return MAPE, MSE

# Function for giving us the input and output based on the selected output


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
# K-fold


def CrossOver(index):
    ScoresMAPE = []
    ScoresMSE = []
    for CrossCount in range(0, K_fold):
        [train_dataset, test_dataset, train_labels,
         test_labels] = Output_moddel_Data(CrossCount)
        mape, mse = Model(train_dataset, train_labels,
                          test_dataset, test_labels, index)

        ScoresMAPE.append(mape)
        ScoresMSE.append(mse)
        print('MAPE=', ScoresMAPE)
        print('MSE=', ScoresMSE)
    return np.mean(ScoresMAPE), np.mean(ScoresMSE)


CrossMape = []
CrossMSE = []
index = []
q1 = len(Dict[Feature]['kernel'])
q2 = len(Dict[Feature]['gamma'])
q3 = len(Dict[Feature]['C'])
q4 = len(Dict[Feature]['epsilon'])
# Defining the optimization treashold parameters
MapeOpt = 1000000
TreasholdCount = 5
TreasureholdAccuracy = .5
count = 0
k = 0
m = 0
z = 0

totMAPE = []
totMSE = []
size = q1*q2*q3*q4
print('Size=', size)
for CountParam in range(0, size):
    index = [choice(range(q1)), choice(range(q3)),
             choice(range(q2)), choice(range(q4))]
    count = count+1
    mape, mse = CrossOver(index)
    print("One crossOver")
    if MapeOpt-mape > TreasureholdAccuracy:
        count = 0
        z = z+1
        TreasholdCount = 10*z+TreasholdCount
    if mape < MapeOpt:
        print("diff=", MapeOpt-mape)
        MapeOpt = mape
        totMAPE.append(MapeOpt)
        totMSE.append(mse)
        m = m+1
        print("mape=", mape)
        print("")
        print("mse=", mse)
        print("")
        bestParam = index
        print("Model Number=", bestParam)
    if count > TreasholdCount:
        break

print("")
print("Resetting the counter=", z)
print("Number of iteration MAPE reduces=", m)
print("Number of iteration MAPE reduces without resettin=", m-z)
print("Opt Iteration=", z+count)
print("Treashold Count=", TreasholdCount)
print("Count=", count)
Info = {"MAPE": totMAPE, "MSE": totMSE, "bestParam": bestParam}
with open(Metrcis, 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, Info.keys())
    w.writeheader()
    w.writerow(Info)

print("mape Total=", totMAPE)

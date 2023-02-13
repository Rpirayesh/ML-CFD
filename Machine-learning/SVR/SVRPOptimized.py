# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:08:18 2020

@author: rpira
"""


from __future__ import absolute_import, division, print_function, unicode_literals
#group rank and group size and type of output
import pandas as pd
from sklearn.svm import SVR
import math
import numpy as np
from random import choice 
import csv
####### data for crossfold, feature, portion,K_fold
dataset= pd.read_csv('DataI300Q7893.csv') # read data set using pandas
Feature='P'
print('Output=',Feature)
Portion=1
K_fold=3
Metrcis=Feature+'MAPEmse.csv'
### Defining the dictionary

Dict={"kernel":['linear','poly', 'rbf', 'sigmoid'],"gamma":[1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],"C":[1, 10, 100, 1000, 10000]}

## Making the GP
kernel=Dict['kernel']
### Defining MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def Model(train_dataset,train_labels,test_dataset,test_labels,index):
    
    regressor = SVR(kernel=Dict['kernel'][index[0]],C=Dict['C'][index[1]], gamma = Dict['gamma'][index[2]], epsilon = 0.01)
    regressor.fit(train_dataset,train_labels)
    y_pred=regressor.predict(test_dataset)
    MSE = ((y_pred-test_labels)**2).mean()
    MAPE=mean_absolute_percentage_error(test_labels, y_pred)
    return MAPE, MSE

### Function for giving us the input and output based on the selected output
def Output_moddel_Data(CrossCount):
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

    pp=len(InputData)
    counter=math.floor(pp/K_fold)    
    ##### Making the data
    IndEnd=(CrossCount+1)*counter
    IndBeging=CrossCount*counter
    test_dataset = InputData[int(IndBeging):int(IndEnd)]
    train_dataset = InputData.drop(test_dataset.index)    
    test_labels = Output[int(IndBeging):int(IndEnd)]
    train_labels = Output.drop(test_labels.index)
    return train_dataset,test_dataset,train_labels,test_labels
#### K-fold
def CrossOver(index):
    ScoresMAPE=[]
    ScoresMSE=[]
    for CrossCount in range(0,K_fold):
        train_dataset,test_dataset,train_labels,test_labels=Output_moddel_Data(CrossCount)
        mape, mse=Model(train_dataset,train_labels,test_dataset,test_labels,index)

        ScoresMAPE.append(mape)
        ScoresMSE.append(mse)
        print('MAPE=',ScoresMAPE)
        print('MSE=',ScoresMSE)
    return np.mean(ScoresMAPE),np.mean(ScoresMSE)
CrossMape=[]
CrossMSE=[]
index=[]
q1=len(Dict['kernel'])
q2=len(Dict['gamma'])
q3=len(Dict['C'])
## 
for hyp in range(0,5):
    index=[choice(range(q1)),choice(range(q3)),choice(range(q2))]
    print('Index=',index)
    ScoresMAPE,ScoresMSE=CrossOver(index)
    CrossMape.append(ScoresMAPE)         
    CrossMSE.append(ScoresMSE)
    print('Giving the mean')
    print('MAPE mean=',CrossMape)
    print('MSE mean=',CrossMSE)
Info={"MAPE":CrossMape,"MSE":CrossMSE}
with open(Metrcis, 'w') as f:  # You will need 'wb' mode in Python 2.x
     w = csv.DictWriter(f, Info.keys())
     w.writeheader()
     w.writerow(Info)
print('Output=',Feature)

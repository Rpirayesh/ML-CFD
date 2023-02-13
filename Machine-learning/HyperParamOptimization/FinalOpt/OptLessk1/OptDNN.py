# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
"""

from numpy.random import randn
from random import choice
from  OPTFinal import * # Importin the optimization objective function
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv


## Making the model and the data
Portion=1
Feature='P'
InputData,Output,Model=Output_moddel_Data(Portion, Feature)

## Defining the k-fold and 
K_fold=3

## Definig a dummy function to make sure the Randomized search is working
c=np.array(list(range(-1000,1000,10)))
values=c**2

## Defining the optimization treashold parameters
MapeOpt=100
TreasholdCount=100
TreasureholdAccuracy=.2
count=0
k=0
m=0
z=0
# Cross over

def CrossOver(K_fold,InputData,Output,ModelInfo):
    ScoresMAPE=[]
    ScoresMSE=[]
    for CrossCount in range(0,K_fold):
        mse,mape =compile_model(CrossCount,K_fold,InputData,Output,ModelInfo)
        ScoresMAPE.append(mape)
        ScoresMSE.append(mse)
    CrossMape=np.mean(ScoresMAPE)
    CrossMSE=np.mean(ScoresMSE)
    return CrossMape,CrossMSE

## Save the optimal model
ModelFile=Feature+'OptModel.obj'
Metrcis=Feature+'OptMAPEmse.csv'
ModelP = open(ModelFile, 'wb')

totMAPE=[]
totMSE=[]
for CountParam in range(0,len(Model)):
    count=count+1
#    ModelInfo=Model[CountParam]
    mape, mse=CrossOver(K_fold,InputData,Output,Model[CountParam])
    print("One crossOver")
    if MapeOpt-mape>TreasureholdAccuracy:
        count=0
        z=z+1
        TreasholdCount=10*z+TreasholdCount
    if mape<MapeOpt:
        print("diff=",MapeOpt-mape)
        MapeOpt=mape
        totMAPE.append(MapeOpt)
        totMSE.append(mse)
        m=m+1
        print("mape=",mape)
        print("")
        print("mse=",mse)
        print("")
#        ModelInfo=Model[CountParam]
        ModelP = open(ModelFile, 'wb')
        bestParam=CountParam
        print("Model Number=",bestParam)
        pickle.dump(Model[CountParam], ModelP)
        print("bestModel",Model[CountParam])
        
    
    if count>TreasholdCount:
       
        break
print("")
print("Resetting the counter=",z)
print("Number of iteration MAPE reduces=",m)
print("Number of iteration MAPE reduces without resettin=",m-z)
print("Opt Iteration=",z+count)
print("Treashold Count=",TreasholdCount)
print("Count=",count)
Info={"MAPE":totMAPE,"MSE":totMSE,"bestParam":bestParam}
with open(Metrcis, 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, Info.keys())
    w.writeheader()
    w.writerow(Info)
#np.savetxt(Metrcis,[totMAPE,totMSE,float(bestParam)] )
#np.savetxt('ModelNumber.csv',bestParam)

print("mape Total=",totMAPE)


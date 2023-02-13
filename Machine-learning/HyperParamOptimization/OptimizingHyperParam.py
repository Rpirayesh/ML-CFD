# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
"""

from OptimizedFunction import compile_model
import pickle

## Importing the database for the combination of hyper parameters 
filehandler = open('filename_pi.obj', 'rb') 
ModelInfo = pickle.load(filehandler)

## Initial values
count=0
tot=[]
MapeOpt=100
TreasureholdAccuracy=1
TreasholdCount=1000
### Finding the minimum mape with 2 treashhold with radomized search
for i in range(0,len(ModelInfo)):
    count=count+1
    saved_model, mape  =compile_model(ModelInfo[i])
    print("Mape=", mape)
    ### Checking if the difference between the current and previous minimum mape
    ### is bigger than the desired on, then restart the counter. So we have 2 treashholdhere
    if MapeOpt-mape>TreasureholdAccuracy:
        count=0
    ## Choose the min mape
    if mape<MapeOpt:
        MapeOpt=mape    
        ## Breacking out of the optimization based on the treashhold
    if count>TreasholdCount:
        break

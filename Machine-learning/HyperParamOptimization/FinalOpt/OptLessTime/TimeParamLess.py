

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
import pickle
from random import shuffle
import random

## Make the dictionary for the values
ModelInfo={}
ModelInfo['Nerouns_L1']=[10,200,2000]
ModelInfo['Nerouns_L2']=[10,100,1000]
ModelInfo['Nerouns_L3']=[10,200,2000]



ModelInfo['Layers']=[2]

ModelInfo['Dropout_Value_L1']=[0]
ModelInfo['Dropout_Value_L2']=[0]
ModelInfo['Dropout_Value_L3']=[0]

ModelInfo['Reguralization_L1']=[0.001,0.01,0.1]
ModelInfo['Reguralization_L2']=[0.001,0.01,0.1]
ModelInfo['Reguralization_L3']=[0.001,0.01,0.1]


ModelInfo['kernel_constraint_L1']=[3,5,10]
ModelInfo['kernel_constraint_L2']=[3,5,10]
ModelInfo['kernel_constraint_L3']=[3,5,10]

ModelInfo['Activation_Method']=['relu','sigmoid']
ModelInfo['Epochs']=[100]
ModelInfo['Batches']=[32]
ModelInfo['Patience']=[100]
ModelInfo['optimizer']=['Adam','Nadam']

ModelInfo['W_Initialization_Method_L1']=[keras.initializers.glorot_uniform(seed=None)]
ModelInfo['W_Initialization_Method_L2']=[keras.initializers.glorot_uniform(seed=None)]
ModelInfo['W_Initialization_Method_L3']=[keras.initializers.glorot_uniform(seed=None)]

TotalModelInfo=[]
ModelInfoMade={}
#TimeParamF = open('TimeParam.obj', 'wb')
#pickle.dump(TotalModelInfo, TimeParamF)
g=0
x=0
ratio=0.2			## ratio of choosing random samples from the made database
num=1000000  ## number of procceed before saving the data
SampleQuantity=num*ratio
for i1 in (ModelInfo['Nerouns_L1']):
    for i2 in (ModelInfo['Nerouns_L2']):
        for i3 in (ModelInfo['Nerouns_L3']):
            
            for j in (ModelInfo['Layers']):
                
                for ll1 in (ModelInfo['Dropout_Value_L1']):
                    for ll2 in (ModelInfo['Dropout_Value_L2']):
                        for ll3 in (ModelInfo['Dropout_Value_L3']):
                            
                            for m1 in ModelInfo['Reguralization_L1']:
                                for m2 in ModelInfo['Reguralization_L2']:
                                    for m3 in ModelInfo['Reguralization_L3']:
                                        
                                        for n1 in ModelInfo['kernel_constraint_L1']:
                                            for n2 in ModelInfo['kernel_constraint_L2']:
                                                for n3 in ModelInfo['kernel_constraint_L3']:
                                                    
                                                    for o in ModelInfo['Activation_Method']:
                                                        for p in ModelInfo['Epochs']:
                                                            for q in ModelInfo['Batches']:
                                                                for r in ModelInfo['optimizer']:
                                                                    
                                                                     for s1 in ModelInfo['W_Initialization_Method_L1']:
                                                                         for s2 in ModelInfo['W_Initialization_Method_L2']:
                                                                             for s3 in ModelInfo['W_Initialization_Method_L3']:
                                                                                 
                                                                                 ModelInfoMade['Nerouns']=[i1,i2,i3]
                                                                                 ModelInfoMade['Layers']=[j]
                                                                                 ModelInfoMade['Dropout_Value']=[ll1,ll2,ll3]
                                                                                 ModelInfoMade['Reguralization']=[m1,m2,m3]
                                                                                 
                                                                                 ModelInfoMade['kernel_constraint']=[n1,n2,n3]
                                                                                 ModelInfoMade['Activation_Method']=[o]
                                                                                 ModelInfoMade['Epochs']=[p]
                                                                                 ModelInfoMade['Batches']=[q]
                                                                                 ModelInfoMade['optimizer']=[r]
                                                                                 ModelInfoMade['Patience']=[100]
                                                                                 ModelInfoMade['W_Initialization_Method']=[s1,s2,s3]
#										 print('W_Initialization_Method=',ModelInfoMade['W_Initialization_Method'])
                                                                                 
                                                                                 TotalModelInfo.append(ModelInfoMade.copy())

shuffle(TotalModelInfo)
print("Length=",len(TotalModelInfo))
TimeParamF = open('TimeParam.obj', 'wb')
pickle.dump(TotalModelInfo, TimeParamF)

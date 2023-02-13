

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

## Make the dictionary for the values
ModelInfo={}
ModelInfo['Nerouns_L1']=[160,2560]
ModelInfo['Nerouns_L2']=[40,200,1000]
ModelInfo['Nerouns_L3']=[30,10,200]
ModelInfo['Nerouns_L4']=[10,60,200]

ModelInfo['Layers']=[2,3,4]

ModelInfo['Dropout_Value_L1']=[0]
ModelInfo['Dropout_Value_L2']=[0]
ModelInfo['Dropout_Value_L3']=[0]
ModelInfo['Dropout_Value_L4']=[0]

#ModelInfo['Reguralization_L1']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]
#ModelInfo['Reguralization_L2']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]
#ModelInfo['Reguralization_L3']=[l1(0.1),l2(0.2),l2(0.1),l1(0.2),l2(0.3),l2(0.4)]

ModelInfo['Reguralization_L1']=[0.001,0.01,0.1]
ModelInfo['Reguralization_L2']=[0.001,0.01,0.1]
ModelInfo['Reguralization_L3']=[0.001,0.01,0.1]
ModelInfo['Reguralization_L4']=[0.001,0.01,0.1]

ModelInfo['kernel_constraint_L1']=[3,5]
ModelInfo['kernel_constraint_L2']=[3,5]
ModelInfo['kernel_constraint_L3']=[3,5]
ModelInfo['kernel_constraint_L4']=[3,5]

ModelInfo['Activation_Method']=['relu','sigmoid']
ModelInfo['Epochs']=[50]
ModelInfo['Batches']=[32]
ModelInfo['optimizer']=['Nadam','Adam']

ModelInfo['W1']=[keras.initializers.glorot_uniform(seed=None)]
ModelInfo['W2']=[keras.initializers.glorot_uniform(seed=None)]
ModelInfo['W3']=[keras.initializers.glorot_uniform(seed=None)]
ModelInfo['W4']=[keras.initializers.glorot_uniform(seed=None)]

TotalModelInfo=[]
ModelInfoMade={}
#EnergyParamF = open('EnergyParam.obj', 'wb')
#pickle.dump(TotalModelInfo, EnergyParamF)
PParamF = open('PParam.obj', 'wb')
pickle.dump(TotalModelInfo, PParamF)
k=0
g=0
#with open('picke.txt', 'ab') as fi:
with open('ErrorParamF.obj', 'a') as ErrorParamF:
	for i1 in (ModelInfo['Nerouns_L1']):
	    for i2 in (ModelInfo['Nerouns_L2']):
                for i3 in (ModelInfo['Nerouns_L3']):
                    for i4 in (ModelInfo['Nerouns_L4']):
		    
                        for j in (ModelInfo['Layers']):
				
                            for ll1 in (ModelInfo['Dropout_Value_L1']):
                                for ll2 in (ModelInfo['Dropout_Value_L2']):
                                    for ll3 in (ModelInfo['Dropout_Value_L3']):
                                        for ll4 in (ModelInfo['Dropout_Value_L4']):
				            
                                             for m1 in ModelInfo['Reguralization_L1']:
                                                 for m2 in ModelInfo['Reguralization_L2']:
                                                     for m3 in ModelInfo['Reguralization_L3']:
                                                            for m4 in ModelInfo['Reguralization_L4']:
				                        
                                                                for n1 in ModelInfo['kernel_constraint_L1']:
                                                                    for n2 in ModelInfo['kernel_constraint_L2']:
                                                                        for n3 in ModelInfo['kernel_constraint_L3']:
                                                                            for n4 in ModelInfo['kernel_constraint_L4']:
                                                                                for r in ModelInfo['optimizer']:
                                                                                    for o in ModelInfo['Activation_Method']:

				                                                                 
                                                                                        ModelInfoMade['Nerouns']=[i1,i2,i3,i4]
                                                                                        ModelInfoMade['Layers']=[j]
                                                                                        ModelInfoMade['Dropout_Value']=[ll1,ll2,ll3,ll4]
                                                                                        ModelInfoMade['Reguralization']=[m1,m2,m3,m4]
                                                                                        ModelInfoMade['kernel_constraint']=[n1,n2,n3,n4]
                                                                                        ModelInfoMade['Activation_Method']=[o]
                                                                                        ModelInfoMade['Epochs']=[50]
                                                                                        ModelInfoMade['Batches']=[32]
                                                                                        ModelInfoMade['Patience']=[50]
                                                                                        ModelInfoMade['optimizer']=[r]
                                                                                        ModelInfoMade['W_Initialization_Method']=[ModelInfo['W1'],ModelInfo['W2'],ModelInfo['W3'],ModelInfo['W4']]
                                                                                        TotalModelInfo.append(ModelInfoMade.copy())

shuffle(TotalModelInfo)
print("Length=",len(TotalModelInfo))
PParamF = open('PParam.obj', 'wb')
pickle.dump(TotalModelInfo, PParamF)
				                                                                 


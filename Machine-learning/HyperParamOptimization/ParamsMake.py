

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

## Make the dictionary for the values
ModelInfo={}
ModelInfo['Nerouns_L1']=[200,80,400]
ModelInfo['Nerouns_L2']=[200,80,400]
ModelInfo['Nerouns_L3']=[200,80,400]

ModelInfo['Layers']=[1,2,3]

ModelInfo['Dropout_Value_L1']=[0,0.2,0,4,0.8]
ModelInfo['Dropout_Value_L2']=[0,0.2,0,4,0.8]
ModelInfo['Dropout_Value_L3']=[0,0.2,0,4,0.8]

ModelInfo['Reguralization_L1']=[l1(0.1),l2(0.2),l2(0.1)]
ModelInfo['Reguralization_L2']=[l1(0.1),l2(0.2),l2(0.1)]
ModelInfo['Reguralization_L3']=[l1(0.1),l2(0.2),l2(0.1)]

ModelInfo['Reguralization_L1']=[l1(0.1),l2(0.2),l2(0.1)]
ModelInfo['Reguralization_L2']=[l1(0.1),l2(0.2),l2(0.1)]
ModelInfo['Reguralization_L3']=[l1(0.1),l2(0.2),l2(0.1)]

ModelInfo['kernel_constraint_L1']=[max_norm(3),max_norm(5)]
ModelInfo['kernel_constraint_L2']=[max_norm(3),max_norm(5)]
ModelInfo['kernel_constraint_L3']=[max_norm(3),max_norm(5)]

ModelInfo['Activation_Method']=['relu','sigmoid']
ModelInfo['Epochs']=[700,1200]
ModelInfo['Batches']=[32,64]
ModelInfo['optimizer']=['Nadam','Adam']

ModelInfo['W_Initialization_Method_L1']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]
ModelInfo['W_Initialization_Method_L2']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]
ModelInfo['W_Initialization_Method_L3']=[keras.initializers.glorot_uniform(seed=None),keras.initializers.lecun_normal(seed=None)]

TotalModelInfo=[]
ModelInfoMade={}
k=0
for i1 in (ModelInfo['Nerouns_L1']):
            
            for j in (ModelInfo['Layers']):
                
                for ll1 in (ModelInfo['Dropout_Value_L1']):
                            
                            for m1 in ModelInfo['Reguralization_L1']:
                                        
                                        for n1 in ModelInfo['kernel_constraint_L1']:
                                                    
                                                    for o in ModelInfo['Activation_Method']:
                                                        for p in ModelInfo['Epochs']:
                                                            for q in ModelInfo['Batches']:
                                                                for r in ModelInfo['optimizer']:
                                                                    
                                                                     for s1 in ModelInfo['W_Initialization_Method_L1']:
                                                                                 
                                                                                 ModelInfoMade['Nerouns']=[i1]
                                                                                 ModelInfoMade['Layers']=[j]
                                                                                 ModelInfoMade['Dropout_Value']=[ll1]
                                                                                 ModelInfoMade['Reguralization']=[m1]
                                                                                 
                                                                                 ModelInfoMade['kernel_constraint']=[n1]
                                                                                 ModelInfoMade['Activation_Method']=[o]
                                                                                 ModelInfoMade['Epochs']=[p]
                                                                                 ModelInfoMade['Batches']=[q]
                                                                                 ModelInfoMade['optimizer']=[r]
                                                                                 ModelInfoMade['W_Initialization_Method']=[s1]
                                                                                 
                                                                                 TotalModelInfo.append(ModelInfoMade)
                                                                                 k=k+1
                                                                                 print(k)
                                                    
                                        
                
                    
                                   
                                        
                            
            
        
            
    
    
                                        
                            
        
        
    
file_pi = open('filename_pi.obj', 'wb') 
pickle.dump(TotalModelInfo, file_pi)
#filehandler = open('filename_pi.obj', 'rb') 
#object = pickle.load(filehandler)


#import math 
#object_pi = math.pi 
#file_pi = open('filename_pi.obj', 'wb') 
#pickle.dump(object_pi, file_pi)


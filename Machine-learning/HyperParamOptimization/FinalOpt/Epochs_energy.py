# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:15:13 2019

@author: rpira
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.models import load_model

import math
MapeBench=100 
steps=4

####### data for crossfold
dataset= pd.read_csv('DataI300Q7893W.csv') # read data set using pandas
#np.random.shuffle(dataset.values)
#dataset.sample(frac=1)
InputData=dataset.copy()

#Split features from labels
#InputData.pop('P')
#InputData.pop('D')
#InputData.pop('Time')
InputData.pop('Error')
OutputData=InputData.pop('Energy')
#
## Normalization Training
def norm(x,stats):
  return (x - stats['mean']) / stats['std']


                        ## build the model
def build_model():
    N0=2560
    k0=layers.Dense(N0, activation=tf.nn.relu, input_shape=[10])
    N1=160
    k1=layers.Dense(N1, activation=tf.nn.relu)
    N2=160
    k2=layers.Dense(N2, activation=tf.nn.relu)
    kend=layers.Dense(1)
    k=[k1]+[k2]
    m=[k0]+k+[kend]
    model = keras.Sequential(m)
                        
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_absolute_percentage_error',
    optimizer=optimizer,
    metrics=['mean_squared_error','mean_absolute_percentage_error'])
    return model
model = build_model()
#
#

             
def make_model(normed_train_data,train_labels,test_labels,normed_test_data,Pat):
        model = build_model()
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=Pat)
        
        mc = ModelCheckpoint(filepath='best_model_Energy.h5', monitor='val_loss', verbose=0, save_best_only=True)                     
        model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                                            validation_split = 0.2, verbose=0, callbacks=[early_stop,mc])
        saved_model = load_model('best_model_Energy.h5')
        loss, mse, mape = saved_model.evaluate(normed_test_data, test_labels, verbose=0)
        return mape
    
K_fold=4


#shuffle(InputData)

pp=len(InputData)
counter=math.floor(pp/K_fold)

ScoresEpochs=[]
for f in range(1,5):
	EPOCHS=f*50
	ScoresPatinece=[]
	for j in range(1,4):
	    Scores=[]
	    Pat=j*50
	    for i in range(K_fold):
	    
	    
	    ##### Making the data
	    
	    	IndEnd=(i+1)*counter
	   	IndBeging=i*counter
	#    print(IndEnd)
	    	test_dataset = InputData[int(IndBeging):int(IndEnd)]
	    	train_dataset = InputData.drop(test_dataset.index)
	    
	    
	    	train_stats = train_dataset.describe()
	    	train_stats = train_stats.transpose()
	    
	    	test_stats = test_dataset.describe()
	    	test_stats = train_stats.transpose()
	    
	    	test_labels = OutputData[int(IndBeging):int(IndEnd)]
	    	train_labels = OutputData.drop(test_labels.index)
	    ### Calling the normalized data
	    	normed_train_data = norm(train_dataset,train_stats)
	    	normed_test_data = norm(test_dataset,train_stats)  
	    	mape=make_model(normed_train_data,train_labels,test_labels,normed_test_data,Pat)
	    	Scores.append(mape)
		print("Scores=",(Scores),",Mean of Scores=",np.mean(Scores),",Std=",np.std(Scores))
		
	    ScoresPatinece.append(np.mean(Scores))    
	    
	ScoresEpochs.append((ScoresPatinece))
	print("ScoresEpochs=",(ScoresEpochs))
	np.savetxt("ScoresPatineceEnergyLowMSE.csv",ScoresEpochs )
	
	
	    

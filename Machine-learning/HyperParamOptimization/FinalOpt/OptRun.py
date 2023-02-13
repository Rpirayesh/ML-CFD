# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
"""

from numpy.random import randn
from random import choice
from OptimizedFunction import compile_model
import pickle
import matplotlib.pyplot as plt
filehandler = open('filename_pi.obj', 'rb') 
import numpy as np

ModelInfo = pickle.load(filehandler)

#values = randn(1000)
c=np.array(list(range(-1000,1000,10)))
values=c**2
#saved_model, mape  =compile_model(ModelInfo[110])
MapeOpt=10000000000000
TreasholdCount=50
TreasureholdAccuracy=10
count=0
k=0
m=0
z=0
#

tot=[]
for i in range(0,np.shape(c)[0]):
    count=count+1
#    saved_model, mape  =compile_model(ModelInfo[i])
    mape=choice(values)
    if MapeOpt-mape>TreasureholdAccuracy:
        count=0
        z=z+1
        print("diff=",MapeOpt-mape)
    if mape<MapeOpt:
        print("diff=",MapeOpt-mape)
        MapeOpt=mape
        tot.append(MapeOpt)
        m=m+1
#        print(mape)
        
    
    if count>TreasholdCount:
        break
print("z=",z)
print("m=",m)

#for i in range(0,np.shape(c)[0]):
##    k=k+1
#    count=count+1
##    saved_model, mape  =compile_model(ModelInfo[i])
#    mape=values[i]
#    print("mape=",mape)
#    if mape<MapeOpt:
#        MapeOpt=mape
#        k=k+1
#        tot.append(MapeOpt)
#        print("MinMape=",mape)
#        count=0        
#    if count>TreasholdCount:
#        break
print("mape=",tot)
        
plt.plot(values)
plt.plot(tot)
#plt.xlabel('Data ')
#plt.ylabel('Predictions [Energy]')
plt.figure() 
plt.plot(tot)
#plt.xlabel('True Values [Energy]')
#plt.ylabel('Predictions [Energy]')
#plt.plot()

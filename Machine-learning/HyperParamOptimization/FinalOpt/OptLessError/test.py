import pickle
from  OPTFinal import *
Portion=1
Feature='Error'
InputData,Output,Model=Output_moddel_Data(Portion, Feature)
#filehandler = open('DataBaseR.obj', 'rb') 
#TimeParam=pickle.load(filehandler)

CountParam=0
CrossCount=3
K_fold=3
mse, mape =compile_model(CrossCount,K_fold,InputData,Output,Model[CountParam])
#print(len(Model))
ModelInfo=Model[CountParam]
print("MAPE=",mape)
print("MSE=",mse)
#print("Model=",ModelInfo)

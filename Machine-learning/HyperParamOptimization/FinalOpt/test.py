import pickle
from  OPTFinal import *
Portion=0.5
Feature='Energy'
InputData,Output,Model=Output_moddel_Data(Portion, Feature)
#filehandler = open('DataBaseR.obj', 'rb') 
#TimeParam=pickle.load(filehandler)

CountParam=2004
CrossCount=3
K_fold=4
saved_model, mape =compile_model(CountParam,CrossCount,K_fold,InputData,Output,Model)
#print(len(Model))
print(mape)

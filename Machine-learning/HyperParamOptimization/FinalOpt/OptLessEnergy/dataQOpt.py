import pickle
from  OPTFinal import *
import csv
Feature='Energy'
#filehandler = open('DataBaseR.obj', 'rb') 
#TimeParam=pickle.load(filehandler)

CountParam=60
CrossCount=0
K_fold=3

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

#mse, mape =compile_model(CrossCount,K_fold,InputData,Output,Model[CountParam])
MAPEQ=[]
MSEQ=[]
PortianV=[]
for i in range(1,21):

    Portion=.05*i
    print('Portion=',Portion)
    InputData,Output,Model=Output_moddel_Data(Portion, Feature)
    ModelInfo=Model[CountParam]
    CrossMape,CrossMSE=CrossOver(K_fold,InputData,Output,ModelInfo)
    MAPEQ.append(CrossMape)
    MSEQ.append(CrossMSE)
    PortianV.append(Portion)
Info={"MAPEQ":MAPEQ,"MSEQ":MSEQ,"Portion":PortianV}
print("MAPE=",MAPEQ)
print("MSE=",MSEQ)
Metrcis=Feature+'DataQOptMAPEmse.csv'
with open(Metrcis, 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, Info.keys())
    w.writeheader()
    w.writerow(Info)
#print(len(Model))

#print("Model=",ModelInfo)

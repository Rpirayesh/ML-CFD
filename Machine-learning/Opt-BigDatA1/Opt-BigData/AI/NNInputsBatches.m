clear all
close all
clc
load('MainReducedDATA')

%% Defining input and output 
ZDataInput=zscore(ReducedDATA(:,[1:6,8])');
Output=zscore(ReducedDATA(:,7)');
%% Design the NN
inputs = ZDataInput;
targets = Output;
%% Param
Devision=80; %%% Number of batches
%%%% changing the neurons 
for i=1:80
   
% Create a Fitting Network
net = fitnet([);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
[net,tr] = train(net,inputs,targets);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
Per(m,k)=performance;
end
% View the Network
% view(net)
i=10:10:100;
figure
plot(i,mean(Per))
xlabel('Number of Layers')
ylabel('Error')
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)




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
%%%% changing the neurons 
k=0;
for i=1:1
    k=k+1;
% Create a Fitting Network
FitnetMatrix=10*ones(1,i);
net = fitnet(FitnetMatrix);

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
i=1:10;
figure
plot(i,mean(Per))
xlabel('Number of Layers')
ylabel('Error')
% View the Network
% view(net)
 
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)




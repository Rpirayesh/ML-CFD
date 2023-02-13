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

for m=1:5
k=0;
    for i=50
    k=k+1;
% Create a Fitting Network
net = fitnet(i);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
net.trainParam.max_fail = 12;
net.trainParam.min_grad=1.00e-10;
[net,tr] = train(net,inputs,targets);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
Per(m,k)=performance;
end
end
% View the Network
% view(net)
% i=10:10:100;
figure
plot(Per)
xlabel('Number of Layers')
ylabel('Error')
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)




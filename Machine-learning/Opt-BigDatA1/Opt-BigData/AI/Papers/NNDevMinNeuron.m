clear all
close all
clc
load('MainReducedDATA')

%% Defining input and output

%%% Energy as the output
ZDataInput=zscore(ReducedDATA(:,[1:6,8])'); %% Euler angles 3, Angular velocities 3, 1 error
% Output=(ReducedDATA(:,7)'); %% 1 energy
%%%%% Controller Parameters and time as the output
Output=(ReducedDATA(:,9:11)'); %% 2 controller, 1 time
%% Design the NN
inputsOld = ZDataInput;
targetsOld = Output;
%% param
Devision=20; %% number of devisions
NumberOFNeurons=5;
NumberOfTest=5;
NDB=round(size(inputsOld,2))/Devision; %%Number Of Data at each Bean
%%%% changing the neurons
for j=1:Devision
    inputs=inputsOld(:,1:j*NDB);
    targets=targetsOld(:,1:j*NDB);
    for m=1:NumberOfTest
        
        % k=0;
        for i=1:NumberOFNeurons
            %     k=k+1;
            % Create a Fitting Network
            net = fitnet(10*2^(i-1));
            
            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 15/100;
            
            % Train the Network
            [net,tr] = train(net,inputs,targets);
            NetS{m,i}=net;
            % Test the Network
            outputs = net(inputs);
            errors = gsubtract(outputs,targets);
            performance = perform(net,targets,outputs);
            Per(m,i)=performance;
        end
    end
    [M,Index]=min(Per);
    PeMin(j,:)=M;
    %%% assigning the minimum net
    
    for u=1:NumberOFNeurons
        NNMin{j,u}=NetS(Index(u),u);
    end
end

save('NetWorkCPT','Per','NNMin')
%% Calculating MAPE
% for w=1:NumberOFNeurons
% for q=1:Devision
%     k=0;
%     MAPEIni=0;
%     for t=1:NDB*q
%         k=k+1;
%     MAPEIni=MAPEIni+abs((targetsOld(t)-NNMin{q,w}{1,1}(inputsOld(:,t)))/(targetsOld(t)));
%     end
%     MAPE(q,w)=MAPEIni/t;
% end
% end

% i=1:Devision;
% figure
% plot(i,PerformanceAve(:,20))
% xlabel('Number of batches')
% ylabel('Error')
% % View the Network
% % view(net)
% i=10:10:500;
% figure
% plot(i,Per)
% xlabel('Number of Layers')
% ylabel('Error')
% save('PerNNNeurons','PerformanceAve')
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)




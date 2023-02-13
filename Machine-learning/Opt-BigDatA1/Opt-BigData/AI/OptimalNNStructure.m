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
NumberOfTest=1;
NDB=round(size(inputsOld,2))/Devision; %%Number Of Data at each Bean
%%%% changing the neurons
for a=Devision:Devision
    inputs=inputsOld(:,1:a*NDB);
    targets=targetsOld(:,1:a*NDB);
    for b=NumberOfTest:NumberOfTest
        
        % k=0;
        for c=1:NumberOFNeurons
            for d=1:NumberOFNeurons
                for e=1:NumberOFNeurons
                    for f=1:NumberOFNeurons
                        for g=1:NumberOFNeurons
                            %     k=k+1;
                            % Create a Fitting Network
                            h=10*2^(c-1);
                            k=10*2^(d-1);
                            l=10*2^(e-1);
                            m=10*2^(f-1);
                            n=10*2^(g-1);
                            %% making the layers hierarically
                            if e==0
                                f=0;
                                g=0;
                                Per(c,d,e,f,g)=5;
                            else if f==0
                                    g=0;
                                    Per(c,d,e,f,g)=5;
                                else if g==0
                                        Per(c,d,e,f,g)=5;
                                    else
                                        %% making the net
                                        net = fitnet([h,k,l,m,n]);
                                        % Set up Division of Data for Training, Validation, Testing
                                        net.divideParam.trainRatio = 70/100;
                                        net.divideParam.valRatio = 15/100;
                                        net.divideParam.testRatio = 15/100;
                                        % Train the Network
                                        [net,tr] = train(net,inputs,targets);
                                        NetS(c,d,e,f,g)=net;
                                        % Test the Network
                                        outputs = net(inputs);
                                        errors = gsubtract(outputs,targets);
                                        performance = perform(net,targets,outputs);
                                        Per(c,d,e,f,g)=performance;
                                        NN{c,d,e,f,g}=net;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
[minval, minidx] = min(A(:));
[ind1,ind2, ind3, ind4, ind5] = ind2sub( size(A), minidx );

save('NetWorkCPT','Per','NN')
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




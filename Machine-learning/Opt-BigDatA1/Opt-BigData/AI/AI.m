clear all
close all
clc
% tic
% %% Loading the optimizaed data from computers and stacking them together
% load('Maindata3t4D8920')
% DATA=MatrixForLearning(1:8920,:);
% load('data3t4D18735')
% DATA=[DATA;MatrixForLearning(8920:18735,:)];
% load('MainData0t1D11424')
% DATA=[DATA;MatrixForLearning(1:11425,:)];
% load('MainData1t2D30044')
% DATA=[DATA;MatrixForLearning(1:30044,:)];
% load('MainData2t3D30572')
% DATA=[DATA;MatrixForLearning(1:30572,:)];
% load('MainData4t5D26757')
% DATA=[DATA;MatrixForLearning(1:26757,:)];
load('NewDataOptimumR')
DATA=MatrixForLearningR(1:3636,[1:7,11:15]);
% save('DATAMade','DATA')
% %% Outlier removal or data cleaning
% %%%%%%% In this part we remove data having more enery and error than the
% %%%%%%% specified ones
% Err=2; %% error limit
% Ene=2; %% Energy limit
k=1;
% tic
for i=1:size(DATA,1)
    if DATA(i,9)<=1 && DATA(i,8)<=2 %% 9 is the eror, 8 is the energy
        ReducedDATAOptim(k,:)=[Q2E(DATA(i,1:4)),DATA(i,5:end)];
        k=k+1;
    end 
end

% toc
save('MainReducedDATAOptim','ReducedDATAOptim')
% ReducedDATAOptim(:,8)=ReducedDATAOptim(:,8)+ReducedDATAOptim(:,9);
csvwrite('DataOptim.csv',ReducedDATAOptim)
% toc
%% load the data and the cleaned data
tic
% load('DATAMade')
% load('MainReducedDATA')
% data=ReducedDATA;
% 
% data=ReducedDATA;
% PercentOfOutliers=1-size(ReducedDATA,1)/size(DATA,1);
%% Z-Normalization
%%%Z-Input, not output
% b=datasample(q,rows,'Replace',false);
% ZDataInput=datasample(zscore(ReducedDATA(:,1:9)),20000,'Replace',false);
% ZDataInput=zscore(data(:,1:6));
% ZDataInput=(data(:,1:8));


% toc
% %% PCA
% n=6;
% PCAData=zeros(size(ZDataInput,1),n);
% C=(ZDataInput'*ZDataInput);
% [V,D]=eig(C);
% 
% for i=1: size(ZDataInput,1)
%    PCAData(i,:)= ZDataInput(i,:)*V(:,1:6);  
% end
% toc
% plot(PCAData(:,1),PCAData(:,2),'.')
% M=[mean(PCAData(:,1)),mean(PCAData(:,2))];
% % plot(PCAData,'.')
% set(gca,'Fontsize',10,'FontName', 'Times New Roman');
% xlabel('First column of data')
% ylabel('Second column of data') 
% grid on
% PD=squareform(pdist(PCAData));
% ED=pdist(ZDataInput);
% distance=squareform(ED);
% rows=size(ZDataInput,1);
% for i=1:rows
%     distance(i,i)=100;
%     PD(i,i)=100;
% end
% for i=1:rows
%     [~,IE(i,:)]=min(distance(i,:));
%    [~,IP(i,:)]=min(PD(i,:));   
%    
% end
% P=0;
% for i=1:rows
%         if IP(i,:)==IE(i,:)
%         P=P+1;
%     end
% end
% PerP=P/rows*100
toc
%% 
tic
%% Deep learning, we used the NNtool to train the deep learning
% input=ZDataInput;
% Output=(data(:,9:11));
% % Mdl = fitrensemble(input,Output(:,1),'Method','LSBoost','NumLearningCycles',100)
% % view(Mdl.Trained{1},'Mode','graph');
% rng(1); % For reproducibility
% t = templateTree('MaxNumSplits',2);
% % Mdl = fitrensemble(input,Output(:,2),'OptimizeHyperparameters','auto',...
% %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
% %     'expected-improvement-plus'));
% % save('OptimizedHpParametersTreeD','Mdl')
% Mdl = fitrensemble(input,Output(:,3),'Learners',t,'CrossVal','on');
% save('CrossValidationEnsemblet','Mdl')
% kflc = kfoldLoss(Mdl,'Mode','cumulative');
% figure;
% plot(kflc);
% ylabel('10-fold cross-validated MSE');
% xlabel('Learning cycle');










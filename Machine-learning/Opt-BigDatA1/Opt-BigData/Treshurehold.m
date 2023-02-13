clear all
clc
close all
load('MatrixCom')
load('MainReducedDATA')
MatrixForLearning=MatrixForLearning(:,[1:7,11:15]);
k=1;

for i=1:size(MatrixForLearning,1)
if MatrixForLearning(i,9)<=1 && MatrixForLearning(i,8)<=2 %% 9 is the eror, 8 is the energy
        ComData(k,:)=[Q2E(MatrixForLearning(i,1:4)),MatrixForLearning(i,5:end)]; %%% ComData represents the data that is going to be compared
        k=k+1;
end
end

% figure(1)
% plot(ComData(:,1),ComData(:,7),'.')
% figure(2)
% plot(ReducedDATA(:,1),ReducedDATA(:,7),'.')

[x,y]=ismember(ComData(:,1:6),ReducedDATA(:,1:6),'rows');
index=find(y~=0);
OptimData=[ComData(index,7)+ComData(index,8),ReducedDATA(y(index),7)+ReducedDATA(y(index),8)]

OptimData2=-(ComData(index,7)+ComData(index,8))+(ReducedDATA(y(index),7)+ReducedDATA(y(index),8))
hist(OptimData2)


hist(ComData(index,7))


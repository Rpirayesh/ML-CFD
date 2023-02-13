clear all
clc
close all
% load MatrixCom1
% D1=MatrixForLearning(1:469,:)
% load MatrixCom2
% D2=MatrixForLearning(1:419,:)
% load MatrixCom3
% D3=MatrixForLearning(1:462,:)
% load MatrixCom4
% D4=MatrixForLearning(1:485,:)
% load MatrixCom5
% D5=MatrixForLearning(1:427,:)
% Data=[D1;D2;D3;D4;D5]
% DataComplete=zeros(size(Data,1),13);
% for i=1:size(Data,1)
%     
%         DataComplete(i,:)=[Data(i,:),optimumTorque(Data(i,7:9))];
%         
% end

        DataComplete(i,:)=[Data(i,:),optimumTorque(Data(i,7:9))];

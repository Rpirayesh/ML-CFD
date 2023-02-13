clear all 
clc
close all
% %% load the data and constraints
% load('OpConstraintsPD')
% load('DataMatrix')
% DATAMatrix=DATAMatrix(randperm(size(DATAMatrix,1)),:);
% save('ShuffledData','DATAMatrix')
% load('ShuffledData')

%% Specifying the features of Gaussian rondam number generator, I believe with this I will have 90% between -1.5 and 1.5
mu=0;
sigma=0.6;

% %% Making the initial condition
% sizeMatrix2=(size(DATAMatrix,1))/5000;
% D2=DATAMatrix(round(sizeMatrix2):round(2*sizeMatrix2),:);

%% Fmincon Initial condition
options = optimoptions(@simulannealbnd,'MaxIterations',300);
% options = optimoptions(@simulannealbnd,'ObjectiveLimit ',0.5);
% options = optimoptions(@patternsearch,'MaxIterations',1);
fun= @optimumTorque; % Function handle to the fitness function

% x0=x
A=[];
B=[];
b=[];
Aeq=[];
beq=[];
lb = [0.0100000000000000,0.0100000000000000,0.00200000000000000]; % Lower bound
ub = [1,1,0.0200000000000000]; % Upper bound
% x0=(lb+ub)/2; 
%% Optimization loop
D2=500;
M1_S=zeros(D2,14);

for i=1:D2
tic

    y0=[E2Q(rand(1,3)*360),normrnd(mu,sigma,[1,3]),-0.0005,0.0006,0.0009];
save('In','y0')
 %% Maaking the Gaussian distribution and random distribution
PropChoisces=[normrnd(5,0.3),rand*5];
Prop=PropChoisces(randi([1,2],1));
save('CoeProp','Prop')
%% Optimization
x0=lb+(ub-lb).*[rand,rand,rand];
[x,fval,exitflag,output]=simulannealbnd(fun,x0,lb,ub,options);

%% Objective functions
Ob=optimumTorqueOF(x);

fval2=Ob(1)+Prop*Ob(2);
%% Making the matrix of data
M1_S(i,:)=[Q2E(y0(1:4)),y0(5:7),x,Prop,Ob,fval,fval2];
% MatrixForLearning(i,:)=[x0,z];
save('MatrixCom1I300FourthRound','M1_S')
toc

end
% %% look at the distributioon of optimizationb
% data=MatrixForLearning(1:i,:);
% k=0;
% for j=1:i
%     if data(j,5)<.5
%         k=k+1;
%         testdata(k,:)=data(j,:);
%     end
% end
save('MatrixCom1I300FourthRound','M1_S')

% plot(DATAMatrix(:,1),MatrixForLearning(:,7),'.')

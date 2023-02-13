clear all 
clc
close all
 load('OpConstraintsPD')
%% making the data
% EulerAngles=linspace(0,360,18);
% AngularVelocities=linspace(-2,2,3);
% sizeM=size(EulerAngles,2)^3*size(AngularVelocities,2)^3; %% size of data 3 Euler angles, 3 angular velocities
% DATAMatrix=zeros(sizeM,6);
% Numerator=0;
% for i=1:size(EulerAngles,2)
%     for j=1:size(EulerAngles,2)
%         for k=1:size(EulerAngles,2)
%             for l=1:size(AngularVelocities,2)
%                 for m=1:size(AngularVelocities,2)
%                     for n=1:size(AngularVelocities,2)
%             Numerator=Numerator+1;
%             DATAMatrix(Numerator,:)=[EulerAngles(i),EulerAngles(j),EulerAngles(k),AngularVelocities(l),AngularVelocities(m),AngularVelocities(n)];
%                     end
%                  end
%             end
%         end
%     end
% end
% 
% DATAMatrix=DATAMatrix(randperm(size(DATAMatrix,1)),:);
% save('DataMatrix','DATAMatrix')
%% Making the initial condition
% to    c
% y1 =[-0.4028    0.0776   -0.8484    0.3594   -2.0353   -0.0788    0.8009   -0.0005    0.0006    0.0009];%% initial condition
% initialQ=y1(end,1:4);
% Eu=Q2E(initialQ);
% % rng(s1); 
% randomNu=rand(1,3);
% NewEu=Eu-2*ones(1,3)*pi/180+4*randomNu*pi/180;
% Qu=E2Q(NewEu);
% NewV=y1(end,5:7)-0.1+0.2*randomNu;
% % y0=[Qu,NewV,10^-3*ones(1,6),y1(end,8:10)];%% when we include the Bw and Bs
% y0=[Qu,NewV,y1(end,8:10)];

%% The loop for optimization

for i
save('In','y0')
%% Fmincon
tic
% options = optimoptions(@patternsearch,'MaxIterations',1)
options = optimoptions(@simulannealbnd,'MaxIterations',1);
% options = optimoptions(@fmincon,'FunctionTolerance',.2,'MaxIterations',1)
% options = optimoptions(@fmincon,'FunctionTolerance',1)
% % options = optimoptions(@fmincon,'OptimalityTolerance',0.5,'OutputFcn',@outfun)
% 
fun= @optimumTorque; % Function handle to the fitness function
x0=(Lo+Up)/2; 
% x0=x
A=[];
B=[];
b=[];
Aeq=[];
beq=[];
lb = Lo; % Lower bound
ub = Up; % Upper bound
% [x,fval] = fmincon(fun,x0,A,B,Aeq,beq,lb,ub,[],options);
% [x,fval] = patternsearch(fun,x0,A,b,Aeq,beq,lb,ub,[],options);
[x,fval] = simulannealbnd(fun,x0,lb,ub,options);
fval
% [x,fval] = simulannealbnd(fun,x0,lb,ub,options);
toc
% function stop = outfun(~,optimValues,~) 
% stop = false;
% if optimValues.fval < .3
%     stop = true;
% end 
% end

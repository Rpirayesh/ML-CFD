clear all
close all
clc
load('MainReducedDATA')
% load('NewDataOptimumR')
%% Data plotted
%
% figure(1)
% plot(ReducedDATA(:,1),ReducedDATA(:,11),'.')
% xlabel('EulerAngle1')
% ylabel('Energy')
% figure(2)
% plot(ReducedDATA(:,2),ReducedDATA(:,7),'.')
% xlabel('EulerAng2')
% ylabel('Energy')
% figure(3)
% plot(ReducedDATA(:,3),ReducedDATA(:,7),'.')
% xlabel('EulerAngle3')
% ylabel('Energy')
% figure(4)
% plot(ReducedDATA(:,4),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity1')
% ylabel('Energy')
% figure(5)
% plot(ReducedDATA(:,5),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity2')
% ylabel('Energy')
% figure(6)
% plot(ReducedDATA(:,6),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity3')
% ylabel('Energy')
% figure(7)
% plot(ReducedDATA(:,8),ReducedDATA(:,7),'.')
% xlabel('Error')
% ylabel('Energy')
%% Histogram
% histogram(ReducedDATA(:,10))
%% Histogram
% histogram(ReducedDATA(:,11))
% hold on
% histogram(ReducedDATA(:,9))
% legend('P','D')
% histogram(ReducedDATA(:,7))
% %% Energy for a single velocity
% figure
% En=find(ReducedDATA(:,4)==-0);
% histogram(ReducedDATA(En,7))
% 
%% Take off data with P:0.5 and D:0.5
Intrerval=find(ReducedDATA(:,10)<0.51 & ReducedDATA(:,10)>0.49);
NoNIntrerval=find(ReducedDATA(:,10)>=0.51 | ReducedDATA(:,10)<=0.49);
PD=ReducedDATA(:,10);
SS=PD(Intrerval);
DataInterval=ReducedDATA(Intrerval,:);
DataNoNInterval=ReducedDATA(NoNIntrerval,:);
figure(1)
h1=histogram(DataInterval(:,7))
h=h1;
hold on
h2=histogram(ReducedDATA(:,7))
figure(2)
histogram(ReducedDATA(:,8))
% csvwrite('DataInterval.csv',DataInterval)
% csvwrite('DataNONInterval.csv',DataNoNInterval)
% figure
%  histogram(ReducedDATA(Intrerval,7))
%  figure
%  histogram(ReducedDATA(NoNIntrerval,7))
%% Data plotted
% figure(1)
% plot(ReducedDATA(:,1),ReducedDATA(:,7),'.')
% xlabel('EulerAngle1')
% ylabel('Energy')
% figure(2)
% plot(ReducedDATA(:,2),ReducedDATA(:,7),'.')
% xlabel('EulerAng2')
% ylabel('Energy')
% figure(3)
% plot(ReducedDATA(:,3),ReducedDATA(:,7),'.')
% xlabel('EulerAngle3')
% ylabel('Energy')
% figure(4)
% plot(ReducedDATA(:,4),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity1')
% ylabel('Energy')
% figure(5)
% plot(ReducedDATA(:,5),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity2')
% ylabel('Energy')
% figure(6)
% plot(ReducedDATA(:,6),ReducedDATA(:,7),'.')
% xlabel('AngularVelocity3')
% ylabel('Energy')
% figure(7)
% plot(ReducedDATA(:,8),ReducedDATA(:,7),'.')
% xlabel('Error')
% ylabel('Energy')


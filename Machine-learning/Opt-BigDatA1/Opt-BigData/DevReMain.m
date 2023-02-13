function [t,y2,u,Error]= DevReMain(fraction,dt)

% tic
% fraction=.02;
% load('Seed')
load('In')
load('orbit_info')
%% solving the equations
%% figuring the initial conditions
Control_time=1*3600/2;

% load('orbit_info')
% load('inertia')%%% loading the desined inertia

%% figuring the released time and angle
%  nu0=pi/20;
%  E0=acos((eF+cos(nu0))/(1+eF*cos(nu0)));
% t_released=(E0-eF*sin(E0))/n;

%% figuring the second phase which is sicence mission
% dtF=1;
% t_end=to+Control_time;
t_end=to-Control_time;
% t2=to-Control_time:dtF:t_end;
%% making the initial condition of quaternions and angular velocities randomely
% y1 =[-0.4028    0.0776   -0.8484    0.3594   -0.0353   -0.0788    0.8009   -0.0005    0.0006    0.0009];%% initial condition
% initialQ=y1(end,1:4);
% Eu=Q2E(initialQ);
% rng(s1);
% randomNu=rand(1,3);
% NewEu=Eu-2*ones(1,3)*pi/180+4*randomNu*pi/180;
% Qu=E2Q(NewEu);
% NewV=y1(end,5:7)-0.1+0.2*randomNu;
% % y0=[Qu,NewV,10^-3*ones(1,6),y1(end,8:10)];%% when we include the Bw and Bs
% y0=[Qu,NewV,y1(end,8:10)];
% RI=[NewEu,NewV];
% Save('IniCondOfDevPha','RI')  

%% main program
% q_minus=y0(1,1:4)';
% p0=eye(9)*10^-6;

% dt=2 ;
% tspan=[to-Control_time-fraction*3600,t_end];
tspan=[to-Control_time-fraction*3600,t_end];
[t,y2]=RK_controlled(tspan-tspan(1),y0,dt);
% toc
u=zeros(max(size(t)),3);
EulerAngles=zeros(max(size(t)),3);
EulerAngles_desired=zeros(max(size(t)),3);

for k=1:size(t,2)
%% figuring the Euler angles with 3-2-1 sequence
Rq=(y2(k,4)^2-y2(k,1:3)*y2(k,1:3)')*eye(3)+2*y2(k,1:3)'*y2(k,1:3)-2*y2(k,4)*skew(y2(k,1:3)');
if abs(Rq(1,3))>1
    Rq(1,3)=sign(Rq(1,3));
end
tetaE=asin(-Rq(1,3));
phiE=atan2(sign(cos(tetaE))*Rq(1,2),sign(cos(tetaE))*Rq(1,1));
sayE=atan2(sign(cos(tetaE))*Rq(2,3),sign(cos(tetaE))*Rq(3,3));
EulerAngles(k,1:3)=[phiE,tetaE,sayE];
EulerAngles2(k,1:3)=Q2E(y2(k,:));
%% Torques,desired quaternions and their errors, desired angular velocity and acceleration
[u(k,:),qc(1,1:4)]=UgradientsF_controlledM_RK(t(k),y2(k,:));

% rr(k,:)=rondqc123'*rondqc123;
%% figuring the desired Euler angles
Rq_desired=(qc(1,4)^2-qc(1,1:3)*qc(1,1:3)')*eye(3)+2*qc(1,1:3)'*qc(1,1:3)-2*qc(1,4)*skew(qc(1,1:3)');
tetaE_d=asin(-Rq_desired(1,3));
phiE_d=atan2(sign(cos(tetaE_d))*Rq_desired(1,2),sign(cos(tetaE_d))*Rq_desired(1,1));
sayE_d=atan2(sign(cos(tetaE_d))*Rq_desired(2,3),sign(cos(tetaE_d))*Rq_desired(3,3));
EulerAngles_desired(k,1:3)=[phiE_d,tetaE_d,sayE_d];

end
%% Error
% xz=EulerAngles*180/pi-ones(size(t2,2),3)*diag(EulerAngles_desired*180/pi');
xz=EulerAngles*180/pi-EulerAngles_desired*180/pi;
Error=zeros(max(size(xz)),1);
for uu=1:max(size(xz))
Error(uu)=(xz(uu,:)*xz(uu,:)')^0.5;
end
%% figure of the Q
% figure
% plot(t/60,y2(:,1:4))
% xlabel('Time, min')
% ylabel('Quaternions')
% axis([0 t(end)/60 -inf inf])
% set(gca,'Fontsize',10,'FontName', 'Times New Roman');
% grid on
% % toc
% %% figure of the quaternions' error
% for i=1:max(size(y2))
%     errorQy2(i)=(y2(i,1:4)*y2(i,1:4)')^0.5;
% end
% figure
% plot((t),errorQy2,'.')
% hold on
% plot(t, 1*ones(1,size(t,2)),'-')
% xlabel('Time, h')
% ylabel('Constraint of quaternions')
% legend('Actual system quaternion constraint','Quaterion constraint=1')
% set(gca,'Fontsize',11,'FontName', 'Times New Roman');
% % axis([0 t(end) 0.9999 1.001])
% axis([0 t(end) 0 inf])
% grid on
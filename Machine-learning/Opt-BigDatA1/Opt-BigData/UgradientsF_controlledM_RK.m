function [u,qc,wc,wcdot,rondqc4,rondqc123]=UgradientsF_controlledM_RK(t,y)
load('inertia')%%% loading the desined inertia
load('orbit_info');
load('noiseGyro')
load('noiseStar')
load('noiseTorque')
load('coeff_control')
Iknown=[I1,0,0;0,I2,0;0,0,I3];
% Ireal=Iknown*(eye(3)+0.2*diag([cos(t);sin(2*t);0.5])*Iknown);
% mue=3.986*10^5;
% eps=0.1*ones(3,1);
% G=10.1*eye(3);
% k=10.1;
% k=1.15;G=0.01*eye(3);eps=0.01;
% k=kk;
% mm=10;
%% equations of motion
%%% These equations are found based on the Euler angles. Here our states
%%% are the Euler angles between the body frame and the orbital
%%% frome.However, in order to find the equations, w is given based on the
%%% w body frame relative to the inertial frame. The equation is:
%%% Iw=-w*Iw+3/R^3*GMO3*IO3+u. The term ''3/R^3*GMO3*IO3'' is because of the gravity
%%% gradient torque on the spacecraft. O3 is the last column of the
%%% rotation matrix of the body frame relative to the orbital frame. We use
%%% the feedback linearization controller to have the error go to zero.
[Toi_f,r_f]=follower_s(t,aL,eL); %% Obtaining the transformation matrix from the orbit frame to the ECI
%%
q=[y(1),y(2),y(3),y(4)]';
w=[y(5),y(6),y(7)]';

q_f=q(1:3,1);
qfskew=[0,-y(3),y(2);y(3),0,-y(1);-y(2),y(1),0]; %% The skew matrix of the Quaternions of the follower
Rbi_f=(y(4)^2-q_f'*q_f)*eye(3)+2*q_f*q_f'-2*y(4)*qfskew;
Rbo_f=Rbi_f*Toi_f'; %% forming the Rbo_f
%% Shaping the rond_q
[qc,wc,wcdot]=formQc_wc_wcdot(t,aL,eL,to);
qcSkew=skew(qc);
Eqc=[qc(4,1)*eye(3)+qcSkew;-qc(1:3,1)'];
rondqc123=Eqc'*q;
rondqc4=q'*qc;
%% sliding mode control
% s=(w-wc)+k*sign(rondqc4)*rondqc123;
% se=(w-wc)+k*rondqc123;
% sb=sbar(s,eps*ones(3,1));
% % sbe=sbar(se,eps);
% uhat=Iknown* (k/2* (abs(rondqc4)*(wc-w) - cross(sign(rondqc4)*rondqc123,(w+wc)) ) + wcdot-G*sb) + cross(w,Iknown*w); 
% % ue=I* (k/2* ((rondqc4)*(wc-w) - cross(rondqc123,(w+wc)) ) + wcdot-G*sbe) + cross(w,I*w); 
% u=(eye(3)-skew(ones(3,1)*Teps))*((eye(3)+diag(Tf*ones(3,1)))*uhat+Tbias*ones(3,1)+sigmaT*randn(3,1));
%% PD controller
uhat=-kd*(1-rondqc123'*rondqc123)*(w-wc)-kp*sign(rondqc4)*rondqc123;
u=(eye(3)-skew(ones(3,1)*Teps))*((eye(3)+diag(Tf*ones(3,1)))*uhat+Tbias*ones(3,1)+sigmaT*randn(3,1));
% u=zeros(3,1);
end

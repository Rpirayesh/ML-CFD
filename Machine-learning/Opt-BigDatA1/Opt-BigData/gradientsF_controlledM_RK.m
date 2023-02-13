function dydt=gradientsF_controlledM_RK(t,y)
load('inertia')%%% loading the desined inertia
load('orbit_info');
load('noiseGyro')
load('noiseStar')
load('noiseTorque')
load('coeff_control')

Iknown=[I1,0,0;0,I2,0;0,0,I3];
Ireal=Iknown*(eye(3)+0.2*diag([cos(t);sin(2*t);0.5])*Iknown);
mue=3.986*10^5;
%  k=kk;
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
% h=[y(14),y(15),y(16)]';
h=[y(8),y(9),y(10)]';
q_f=q(1:3,1);
qfskew=[0,-y(3),y(2);y(3),0,-y(1);-y(2),y(1),0]; %% The skew matrix of the Quaternions of the follower
Rbi_f=(y(4)^2-q_f'*q_f)*eye(3)+2*q_f*q_f'-2*y(4)*qfskew;
Rbo_f=Rbi_f*Toi_f'; %% forming the Rbo_f
gg_f=3*mue/r_f^3*cross(Rbo_f(:,3),Ireal*Rbo_f(:,3)); %% gravity gradient torque on the follower
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
% % % sbe=sbar(se,eps);
% uhat=Iknown* (k/2* (abs(rondqc4)*(wc-w) - cross(sign(rondqc4)*rondqc123,(w+wc)) ) + wcdot-G*sb) + cross(w,Iknown*w); 
% % ue=I* (k/2* ((rondqc4)*(wc-w) - cross(rondqc123,(w+wc)) ) + wcdot-G*sbe) + cross(w,I*w); 
% u=(eye(3)-skew(ones(3,1)*Teps))*((eye(3)+diag(Tf*ones(3,1)))*uhat+Tbias*ones(3,1)+sigmaT*randn(3,1));
%% PD controller
uhat=-kd*(1-rondqc123'*rondqc123)*(w-wc)-kp*sign(rondqc4)*rondqc123;
u=(eye(3)-skew(ones(3,1)*Teps))*((eye(3)+diag(Tf*ones(3,1)))*uhat+Tbias*ones(3,1)+sigmaT*randn(3,1));
% u=zeros(3,1);
%% systems of equations
g=Ireal^-1*(gg_f-cross(w,Ireal*w)+u+Tnoise);%% For omegas
f=0.5*[qfskew+y(4)*eye(3);-y(1),-y(2),-y(3)]*w;%% For Quaternions
% BW=-1/toy_b*y(8:10)'+bwSigma*randn(3,1); %% for gyro bias noise
% BS=-1/toy_s*y(11:13)'+bsSigma*randn(3,1); %% for star camera bias noise
h=-skew(w)*h-u;
% dydt=[f;g;BW;BS;h];
dydt=[f;g;h];
end
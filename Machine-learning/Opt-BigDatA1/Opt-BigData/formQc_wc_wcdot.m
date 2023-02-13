function [qc_bi,wc_bi,wcdot_bi]=formQc_wc_wcdot(t,a,e,to)
%% forming the nu and nudot and nuddot which are in the equations of motion, they have to be tracked by teta 2 for the time control to watch the Crab Nebula
mue=3.986*10^5;
Control_time=360/2;
aF=a;
eF=e;
pF=aF*(1-eF^2);
hF=(mue*pF)^0.5;
n=(mue/(aF^3))^0.5;
MF=n*t;
EFF = keplersolve(MF,eF);
rF=aF*(1-eF*cos(EFF));
cosnuF=(eF-cos(EFF))/(eF*cos(EFF)-1);
sinnuF=aF*(1-eF^2)^0.5 /rF*sin(EFF);

if t<=to
    nuF=atan2(sinnuF,cosnuF);
else
    nuF=atan2(sinnuF,cosnuF)+2*pi;
end
nudF=hF/rF^2;
nudFtest=mue^2*(1+eF*cos(nuF))^2 / hF^3;
nuddF=-2*mue^2/hF^3*(1+eF*cos(nuF))*(eF*nudF*sin(nuF));
%% forming the desired Euler angles and desired Ueler angles dot and desired Ueler angles ddot
t_desired1=0;
td_desired1=0;
tdd_desired1=0;
if t<=to
    t_desired2=-(pi-nuF);
    td_desired2=-nudF;
    tdd_desired2=-nuddF;
else
    t_desired2=(pi-nuF);
   td_desired2=nudF;
   tdd_desired2=nuddF;
end

% if t<to-Control_time+90
td_desired3=0;
% else
%     td_desired3=0;
% end
t_desired3=td_desired3*t;
tdd_desired3=0;
%% forming the qc using 1-2-3 Sequence from LVLH to Fb
c1=cos(t_desired1);
s1=sin(t_desired1);
c2=cos(t_desired2);
s2=sin(t_desired2);
c3=cos(t_desired3);
s3=sin(t_desired3);
R3=[c3,s3,0;-s3,c3,0;0,0,1];
R2=[c2,0,-s2;0,1,0;s2,0,c2];
R1=[1,0,0;0,c1,s1;0,-s1,c1];
Rbl=R3*R2*R1;
%% forming the Rol rotation by the sequence i2=t_desired2 i1=90 i3=-90
i2=t_desired2;
i1=pi/2;
i3=-pi/2;
ci1=cos(i1);
si1=sin(i1);
ci2=cos(i2);
si2=sin(i2);
ci3=cos(i3);
si3=sin(i3);
Ri3=[ci3,si3,0;-si3,ci3,0;0,0,1];
Ri2=[ci2,0,-si2;0,1,0;si2,0,ci2];
Ri1=[1,0,0;0,ci1,si1;0,-si1,ci1];
Rol=Ri3*Ri1*Ri2;
Rbo=Rbl*Rol';
%% Forming the rotation Roi
[Toi,~]=follower_s(t,a,e);
Rbi=Rbo*Toi;
% qcc=rotm2quat(Rbi);
qc_bi4=0.5*(1+trace(Rbi))^0.5;
qc_bi123=1/4/qc_bi4*[Rbi(2,3)-Rbi(3,2);Rbi(3,1)-Rbi(1,3);Rbi(1,2)-Rbi(2,1)];
qc_bi=[qc_bi123;qc_bi4];
%% Forming the wc
wc_bl=[c2*c3,s3,0;-c2*s3,c3,0;s2,0,1]*[td_desired1;td_desired2;td_desired3];
wc_li=Rbl*[0;-nudF;0];
wc_bi=wc_bl+wc_li;
%% Forming the wcdot
G=[-td_desired2*td_desired1*s2*c3-td_desired3*td_desired1*s3*c2+td_desired3*td_desired2*c3-nudF*td_desired1*c1*s2*c3-nudF*td_desired2*c2*s1*c3+nudF*td_desired3*s3*s1*s2+nudF*td_desired1*s1*s3-nudF*c1*td_desired3*c3;...
    td_desired1*td_desired2*s2*s3-td_desired1*td_desired3*c2*c3-td_desired2*td_desired3*s3+nudF*s1*td_desired1*c3+nudF*td_desired3*s3*c1+nudF*td_desired1*c1*s2*s3+nudF*td_desired2*s1*c2*s3+nudF*s1*s2*td_desired3*c3;...
    nudF*td_desired1*c1*c2-nudF*s1*td_desired2*s2];
b=[-nuddF*(s1*s2*c3+c1*s3);-nuddF*(c1*c3-s1*s2*s3);nuddF*s1*c2];
wcdot_bi=[c2*c3,s3,0;-c2*s3,c3,0;s2,0,1]*[tdd_desired1;tdd_desired2;tdd_desired3]+G+b;

function q=E2Q(input)
%% 3-2-1 sequence
% [phi,teta,say]=input;
phi=input(1);
teta=input(2);
say=input(3);
s1=sin(phi);
c1=cos(phi);
s2=sin(teta);
c2=cos(teta);
s3=sin(say);
c3=cos(say);
R=[c1*c2,s1*c2,-s2;
    -s1*c3+c1*s2*s3,c1*c3+s1*s2*s3,c2*s3;
    s1*s3+c1*s2*c3,-c1*s3+s1*s2*c3,c2*c3];
qc_bi4=0.5*(1+trace(R))^0.5;
qc_bi123=1/4/qc_bi4*[R(2,3)-R(3,2);R(3,1)-R(1,3);R(1,2)-R(2,1)];
q=[qc_bi123;qc_bi4]';
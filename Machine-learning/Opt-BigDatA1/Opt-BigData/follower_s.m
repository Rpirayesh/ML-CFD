function [Toi,r]=follower_s(t,a,e)
%% we solve the E and M to find the transition matrix Tio and to find the distance from earth
load('orbit_info2')
mue=3.986*10^5;
cR=cos(RAAN);
sR=sin(RAAN);
ci=cos(inc);
si=sin(inc);
co=cos(omega);
so=sin(omega);
p=a*(1-e^2);
    M=(mue/a^3)^0.5*t/3600;
E=kepler_E(e,M);
nu=2*atand(tan(E/2)/((1-e)/(1+e))^0.5);
r=p/(1+e*cos(nu));
Toi=[cR*co-sR*so*ci,sR*co+cR*so*ci,so*si;...
    -cR*so-sR*co*ci,-sR*so+cR*co*ci,co*si;...
    sR*si,-cR*si,ci];
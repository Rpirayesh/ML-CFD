function [tWhole,yRK]=RK_controlled(tspanfree,y0,dt) 
t = tspanfree(1); 
s=length(y0);
tWhole=tspanfree(1):dt:tspanfree(2);
m=length(tWhole);
x=zeros(m,s);
% q0=y0(1:4);
% q0=q0(:);
% q0=q0/norm(q0);
% w0=y0(5:8);
% w0=w0(:);
x(1,:)=y0;
for i=1:m-1
    k1 = dt*gradientsF_controlledM_RK(t,x(i,:));
    k2 = dt*gradientsF_controlledM_RK(t+dt/2, x(i,:)+k1'/2);
    k3 = dt*gradientsF_controlledM_RK(t+dt/2, x(i,:)+k2'/2);
    k4 = dt*gradientsF_controlledM_RK(t+dt, x(i,:)+k3');
    x(i+1,:) = x(i,:) + (k1'+2*k2'+2*k3'+k4')/6;
    t = t + dt;
end
yRK=x;
end
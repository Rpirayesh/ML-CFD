function z=optimumTorque(x)
% tic
 load CoeProp %% loading the Gaussian gain for the eeror in the optimization
% toc
% load('OpConstraints')
% load('OpConstraintsPD')
% s1 = rng;
% save('Seed','s1')
%% defining the optimization variables
% x=[1.899,0.226,1]
% kk=x(1);
% G=x(2);
% eps=x(3);
% fraction=x(4);

% kk=1;
% G=0.01;
% eps=0.01;
kp=x(1);
kd=x(2);
fraction=x(3);
% kp=.1;
% kd=.1;
delay=4;
% fraction=.1;
% save('coeff_control','kk','G','eps','fraction');
% load('coeff_control')
save('coeff_control','kp','kd','fraction');
%% initial condition
% teta1=-31.2*pi/180;%% initial condition
% teta2=198.5*1*pi/180;%% initial condition
% teta3=111.5*pi/180;%% initial condition
% tetad1=7.3*pi/180;%% initial condition
% tetad2=5*pi/180;%% initial condition
% tetad3=-8*pi/180;%% initial condition
% fraction=.01;
%% Defining the time step of the solver
MaxPD=max([kp,kd]);
if MaxPD>2 && MaxPD<=3 
    dt=0.08;
    steps=delay/dt;
else if MaxPD>1 && MaxPD<=2
        dt=0.1;
        steps=delay/dt;
    else if MaxPD>0.5 && MaxPD<=1
            dt=0.2;
            steps=delay/dt;
        else if MaxPD>0.1 && MaxPD<=.5
                dt=2;
                steps=delay/dt;
            else if MaxPD<=.1
                    dt=2;
                    steps=delay/dt;
            end
        end
    end
    end
end
 
% if isempty(dt)
%     dt=.05;
% end
% dt=0.14;
%% developoment phase
[t,y,u,Error]= DevReMain(fraction,dt);

% save('Er','t','y','u','Error')
%% finding the divergence of the function
% load('Er')
Div=min([2,abs(y(end,:))]);
if Div==2
    [t,y,u,Error]= DevReMain(fraction,0.05);
    Div=min([2,abs(y(end,:))]);
end

if Div==2 || floor(t(end))+2<floor(fraction*3600)
%% The target function
ob(1)=100;
ob(2)=100;
% z=z(1)+z(2)
else
ob(1)=trapz(t,abs(y(:,5).*u(:,1)))+trapz(t,abs(y(:,6).*u(:,2)))+trapz(t,abs(y(:,7).*u(:,3))); %% energy
ob(2)=1/delay*trapz(t(end-steps:end),abs(Error(end-steps:end,1)));  %% Error 
end
% x;
z=ob;
% Prop=1;
z=ob(1)+Prop*ob(2);
% save('DSolver','delay','steps','dt')
% z=ob;


% if z>10
%     z=2
% end
% save('TargetFunValues','z')
% %% plots
% t=[t1;t2;t3]/60;
% y=[y1;y2;y3];
% figure
% plot(t,[y(:,1),y(:,3),y(:,5)]*180/pi)
% xlabel('time(min)','fontsize',10)
% ylabel(' angles(deg)','fontsize',10)
% legend('teta1','teta2','teta3')
% figure
% plot(t,[y(:,2),y(:,4),y(:,6)]*180/pi)
% xlabel('time(min)','fontsize',10)
% ylabel('andangular velocities (deg/sec)','fontsize',10)
% legend('tetadot1','tetadot2','tetadot3')
% toc
% end
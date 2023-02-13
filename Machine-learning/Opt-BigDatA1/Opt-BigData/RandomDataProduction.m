clear all 
clc
close all
%% Defining the range of change of the angles and Velocities
SeperationEu=18;
SeperationV=3;
%% Specifying the features of Gaussian rondam number generator, I believe with this I will have 90% between -1.5 and 1.5
mu=0;
sigma=0.6;
%% making the data
EulerAngles=linspace(0,360,SeperationEu);
AngularVelocities=linspace(-2,2,SeperationV);
sizeM=size(EulerAngles,2)^3*size(AngularVelocities,2)^3; %% size of data 3 Euler angles, 3 angular velocities
DATAMatrix=zeros(sizeM,6);
Numerator=0;

for i=1:size(EulerAngles,2)
    for j=1:size(EulerAngles,2)
        for k=1:size(EulerAngles,2)
            for l=1:size(AngularVelocities,2)
                for m=1:size(AngularVelocities,2)
                    for n=1:size(AngularVelocities,2)
            Numerator=Numerator+1;
%             RandEU=(EulerAngles(2)-EulerAngles(1))*rand; %%% random number given to Euler Angle to fill the gap between Euler angles values
%             RandV=(AngularVelocities(2)-AngularVelocities(1))*rand; %%% random number given to Angular velocities to fill the gap between Angular velocities
V1Rand=normrnd(mu,sigma);
V2Rand=normrnd(mu,sigma);
V3Rand=normrnd(mu,sigma);
            DATAMatrix(Numerator,:)=[EulerAngles(i)+(EulerAngles(2)-EulerAngles(1))*rand,EulerAngles(j)+(EulerAngles(2)-EulerAngles(1))*rand,EulerAngles(k)+(EulerAngles(2)-EulerAngles(1))*rand,,];
%             DATAMatrix2(Numerator,:)=[EulerAngles(i),EulerAngles(j),EulerAngles(k),AngularVelocities(l),AngularVelocities(m),AngularVelocities(n)];

                    end
                 end
            end
        end
    end
end

DATAMatrixR=DATAMatrix(randperm(size(DATAMatrix,1)),:);
save('DataMatrixRandom','DATAMatrixR')
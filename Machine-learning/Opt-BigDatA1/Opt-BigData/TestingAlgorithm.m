clear all 
clc
close all

load 'MatrixCom'
k=0;
for i=1:size(MatrixForLearning,1)
    if MatrixForLearning(i,8)>2
        k=k+1;
        m(k,:)=MatrixForLearning(i,:);
    end
end
y0=[E2Q(m(1,1:3)),m(1,4:6),-0.0005,0.0006,0.0009];

save('In','y0')

r=optimumTorque(m(1,7:9))

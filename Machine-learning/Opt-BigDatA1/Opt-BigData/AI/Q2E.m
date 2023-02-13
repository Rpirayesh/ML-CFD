function y=Q2E(Input)
Rq=(Input(1,4)^2-Input(1,1:3)*Input(1,1:3)')*eye(3)+2*Input(1,1:3)'*Input(1,1:3)-2*Input(1,4)*skew(Input(1,1:3)');
if abs(Rq(1,3))>1
    Rq(1,3)=sign(Rq(1,3));
end
tetaE=asin(-Rq(1,3));
phiE=atan2(sign(cos(tetaE))*Rq(1,2),sign(cos(tetaE))*Rq(1,1));
sayE=atan2(sign(cos(tetaE))*Rq(2,3),sign(cos(tetaE))*Rq(3,3));
y=[phiE,tetaE,sayE];
function sb=sbar(s,eps)
sb=ones(3,1);
for i=1:3
    if s(i)>eps(i)
        sbb=1;
    else if abs(s(i))<=eps(i)
            sbb=s(i)/eps(i);
        else if s(i)<eps(i)
                sbb=-1;
            end
        end
    end
    sb(i)=sbb;
end
function dphi=solve_dphi()
    dphi0=[0;0;0;0];
    A=[];
    b=[];
    Aeq=[];
    beq=[];
    n=1;
    VLB=[-pi/n;-pi/n;-pi/n;-pi/n];
    VUB=[ pi/n; pi/n; pi/n; pi/n];
    [dphi,~]=fmincon(@min_J,dphi0,A,b,Aeq,beq,VLB,VUB);
end
function hunt_ps=hunt_point(temp_aim_pos,aim_pos,R)
    dv=aim_pos-temp_aim_pos;
    psi=atan(dv(1,2)/dv(1,1));
    p1=[cos(psi+pi),sin(psi+pi)]*R+aim_pos;
    p2=[cos(psi+pi/2),sin(psi+pi/2)]*R+aim_pos;
    p3=[cos(psi),sin(psi)]*R+aim_pos;
    p4=[cos(psi-pi/2),sin(psi-pi/2)]*R+aim_pos;
    hunt_ps=[p1;p2;p3;p4];
end
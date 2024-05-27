function [D,matchs]=distance(uav_pos,hunt_ps,D)
    %i为无人机编号，j为围捕点编号
    for i=1:4
        for j=1:4
            D(i,j)=norm(uav_pos(i,:)-hunt_ps(j,:),2);
        end
    end
    matchs=match(D);
end
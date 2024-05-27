function aim_pos=aim_move(t,t_rate,r)
    aim_pos=r*[cos(t_rate*t),sin(t_rate*t)];
end

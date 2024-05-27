clc
close all
clear
global aim_pos
aim_pos=[ 1, 0];
global uav_pos
uav_pos=(rand(4,2)-0.5)*6;
global temp_aim_pos
temp_aim_pos=aim_pos;
global phi
phi=(rand(4,1)-0.5)*2*pi;
h0=line(NaN,NaN,'marker','o','erasemode','none','color','r');
h1=line(NaN,NaN,'marker','x','erasemode','none','color','b');
h2=line(NaN,NaN,'marker','x','erasemode','none','color','b');
h3=line(NaN,NaN,'marker','x','erasemode','none','color','b');
h4=line(NaN,NaN,'marker','x','erasemode','none','color','b');
w1=line(NaN,NaN,'marker','*','erasemode','none','color','g');
w2=line(NaN,NaN,'marker','*','erasemode','none','color','g');
w3=line(NaN,NaN,'marker','*','erasemode','none','color','g');
w4=line(NaN,NaN,'marker','*','erasemode','none','color','g');
T=360;
global V
V=pi/200;
R=0.2;
D=zeros(4,4);
global matchs
matchs=zeros(4,2);
global hunt_ps
%%
figure(1)
axis([-3,3,-3,3]);
for t=1:T
    %目标位置
    temp_aim_pos=aim_pos;
    aim_pos=aim_move(t,pi/180,1);
    %围捕点位置
    hunt_ps=hunt_point(temp_aim_pos,aim_pos,R);
    %各机器人到各围捕点的距离,i为无人机编号，j为围捕点编号
    %matchs第一列为无人机编号，第二列为围捕点编号
    [D,matchs]=distance(uav_pos,hunt_ps,D);
    %无人机位置
    dphi=solve_dphi();
    phi=phi+dphi;
    uav_pos=uav_pos+V*[cos(phi),sin(phi)];
    %围捕点显示
    set(w1,'xdata',hunt_ps(1,1),'ydata',hunt_ps(1,2));
    hold on
    set(w2,'xdata',hunt_ps(2,1),'ydata',hunt_ps(2,2));
    hold on
    set(w3,'xdata',hunt_ps(3,1),'ydata',hunt_ps(3,2));
    hold on
    set(w4,'xdata',hunt_ps(4,1),'ydata',hunt_ps(4,2));
    hold on
    %目标位置显示
    set(h0,'xdata',aim_pos(1,1),'ydata',aim_pos(1,2));
    hold on
    %1号无人机位置显示
    set(h1,'xdata',uav_pos(1,1),'ydata',uav_pos(1,2));
    hold on
    %2号无人机位置显示
    set(h2,'xdata',uav_pos(2,1),'ydata',uav_pos(2,2));
    hold on
    %3号无人机位置显示
    set(h3,'xdata',uav_pos(3,1),'ydata',uav_pos(3,2));
    hold on
    %4号无人机位置显示
    set(h4,'xdata',uav_pos(4,1),'ydata',uav_pos(4,2));
    pause(0.01);
end
from typing import Dict, List
import numpy as np
from Sim_Class import UGV, UGV_Group, Hunting_Tar, Obstacles
from Plot_figs import plot_dynamic_traces_multi_apollo, plot_dynamic_traces_multi_ParamTest,plot_dynamic_traces_multi_ParamTest_no_apollo
from Dynamic_Assignment import Dynamic_Assignment


N = 3
Maxiter:int = 100
dt:float = 0.5 

# 人工势场参数

# 追捕者内部/追捕者与目标斥力计算半径
R_EP = 0.18

# 目标引力计算半径
R_PP = 100000000

# 引力计算常数
eta_hunt_pts = 0.8

# Pursuer斥力常数
eta_inner = 0.13
# 逃跑车感知半径
R_sensor= 100000

# Evader斥力常数
eta_Tar = 500000000

APF_param = [R_EP, eta_inner, R_sensor, eta_Tar, R_PP, eta_hunt_pts]

# 输出限幅
Tar_theta_acc_bound, Tar_v_acc_bound, Tar_v_bound = 0.1 * np.pi, 0.08, 0.08
Tar_bound = [Tar_theta_acc_bound, Tar_v_acc_bound, Tar_v_bound]

UGV_theta_acc_bound, UGV_v_acc_bound, UGV_v_bound = 0.6 * np.pi, 0.25, 0.25
UGV_bound = [UGV_theta_acc_bound, UGV_v_acc_bound, UGV_v_bound]

# 基于速度生成一个围捕半径
r_hunt = 0.57

# 集群及目标初始化

# 障碍物初始化
# all_Obstacle = Obstacles((600,400), (700,-200), (730,-200), (730,-230), (700,-230))
all_Obstacle = Obstacles((10000,10000))
# 集群初始化参数
kp, ki, kd = 1.5, 0, 0
k_pid = [kp, ki, kd]
ugvs_params_0 = np.array([
    [3, 4, 0.1, 0, 0, k_pid, 0, UGV_bound, dt],
    [-0.2, 3, 0.1, 90, 1, k_pid, 0, UGV_bound, dt],
    [4, -4, 0.1, 90, 7, k_pid, 0, UGV_bound, dt],
    [4, 0, 0.2, 45, 2, k_pid, 0, UGV_bound, dt], 
    [-0.3, 0.7, 0.1, 0, 6, k_pid, 0, UGV_bound, dt]],)
ugvs_params_0 = np.array(ugvs_params_0, dtype=object)

# 目标与集群初始化
Target_0 = Hunting_Tar(Tar_id=0, x=3, y=-4, speed=5, theta=0, r_hunt=r_hunt, Tar_bound=Tar_bound, start_time=0, Tar_value=100, dt=dt)
Group_0 = UGV_Group(group_id=0, ugvs_params=ugvs_params_0)

ugvs_params_1 = np.array([
    [-1, 1, 0.1, 0.2, 3, k_pid, 0, UGV_bound, dt],
    [-2, -0.9, 0.1, 0, 4, k_pid, 0, UGV_bound, dt],
    [-3, 1.2, 0.1, 0, 8, k_pid, 0, UGV_bound, dt],
    [0, -1, 0.1, 0, 5, k_pid, 0, UGV_bound, dt]])
ugvs_params_1 = np.array(ugvs_params_1, dtype=object)
Target_1 = Hunting_Tar(Tar_id=1, x=-5, y=-3, speed=0.06, theta=0, r_hunt=r_hunt, Tar_bound=Tar_bound, start_time=0, Tar_value=100,dt=dt)
Group_1 = UGV_Group(group_id=1, ugvs_params=ugvs_params_1)

Groups_list = [Group_0, Group_1]
Targets_list = [Target_0, Target_1]

# 任务管理器初始化
init_Pairs = {Target_0:Group_0, Target_1:Group_1}
hunted_threshold = 0.25
Max_Path:float = Maxiter*dt*UGV_v_bound
R_dens:float = 1
Tar_Benefit_Params:Dict[str, float] = {'k_dens':1, 'k_val':1}
UGVs_Cost_Params:Dict[str, float] = {'k_dis':0.99, 'k_theta':0.01, 'k_weapon':0}

init_Params = {'Pairs': init_Pairs, 'Other_Params': {'hunted_threshold':hunted_threshold, 'Max_Path':Max_Path, 'R_dens':R_dens, 'Tar_Benefit_Params':Tar_Benefit_Params, 'UGVs_Cost_Params':UGVs_Cost_Params, 'Obstacle':all_Obstacle}}
Task_Manneger = Dynamic_Assignment(init_Params)

Target_2 = Hunting_Tar(Tar_id=2, x=-1, y=5, speed=0.06, theta=0.8, r_hunt=r_hunt, Tar_bound=Tar_bound, start_time=25, Tar_value=100, dt=dt)

for iter in range(0, Maxiter):
    # print(iter)

    # First：态势感知：包括新目标搜索、当前全部UGVs获取

    new_Targets_list = Targets_list 

    if iter >= 25:
        new_Targets_list = Targets_list + [Target_2]
    
    # 0、利用效用函数集中式分配围捕目标
    # flag=1 ：最简单的匈牙利分配方法 
    # flag = 2 :ECBBA算法
    # flag = 3: 贪心算法
    # flag = 4: 合同网协议

    new_Groups_list = Task_Manneger.Target_Assignment(Groups_list, new_Targets_list, iter, flag=4)
    
    # 1、更新UGV群的位置/速度、围捕目标位置/速度
    Task_Manneger.update_all_position()

    # 2、围捕周围的围捕点更新及分配
    # flag = 1 基于固定角度势点
    # flag = 2 基于距离优化势点
    # flag = 3 基于距离、角度优化势点
    # flag = 4 基于合同网的距离、角度优化

    Task_Manneger.update_all_hunt_pts(flag=2)

    # 3、将围捕点赋给对应UGV
    # flag = 1 基于距离的匈牙利分配算法
    # flag = 2 再议
     
    Task_Manneger.update_all_matches(flag=1)

    # 4、根据围捕点生成每个UGV当前参考轨轨迹
    # flag = 
    # 1：直接将目标点轨迹赋给UGV
    # 2: 各类轨迹生成方法
    Task_Manneger.update_all_ref_path(flag=1)

    # 五、依据当前时刻状态进行每个艇速度、加速度计算，同时更新目标的速度、加速度
    
    all_ugv_force_id = Task_Manneger.update_all_APF(APF_param, flag=1)
    
    # # a. 艇群
    # # flag_UGV = 
    # # 1：比例导引率+固定速度
    # # 2: 全部人工势场
    # # 3：人工势场作为局部路径规划器，全局用比例引导率跟，后续用MPC跟 --- 这里的说法参考：冰雪条件下基于改进 MPC 的轨迹跟踪算法研究
    # # 4: 改进的用vector2d计算的人工势场受力

    # # b. 目标：总是根据APF进行逃逸
    # # flag_Tar = 
    # # 1: 人工势场逃跑 version_1.0
    # # 2: 人工势场逃跑 version_2.0
    # # 3: 人工势场逃跑 version_3.0 --- 受到所有人的影响

    Task_Manneger.update_all_acceleration(flag_UGV=4, flag_Tar=3)

# 多目标绘图

# plot_dynamic_traces_multi_ParamTest(new_Groups_list, new_Targets_list, all_Obstacle, r_hunt, dt)

# plot_dynamic_traces_multi_discoupled(new_Groups_list, new_Targets_list, r_hunt, dt)
plot_dynamic_traces_multi_ParamTest_no_apollo(new_Groups_list, new_Targets_list, all_Obstacle, r_hunt, dt, save_path='C:\\Users\\toby\\Desktop\\Dropbox\\Papers\\000 USV-ICUS-Hunt\\test_figs', m=1)
# plot_dynamic_traces_multi_ParamTest(new_Groups_list, new_Targets_list, all_Obstacle, 0.6, dt, save_path='C:\\Users\\toby\\Desktop\\Dropbox\\Papers\\000 USV-ICUS-Hunt\\sim_figs', m=5)
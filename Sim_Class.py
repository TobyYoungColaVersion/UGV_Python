from typing import Dict, List
import math
import numpy as np
import copy
from scipy.optimize import minimize, linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Other_Func import calculate_angle_error, point_on_circle, total_distance, angle_constraint, mid_angle, apollonius_circle, match, get_Inner_Dis, PID, contract_net_angle_assign, vehicle_deployment_optimization
from Artifical_Potential_Field import APF_original, APF_Improved                       

class UGV:
    def __init__(self, x, y, speed, theta, id, k_pid, start_time, bound, sample_time):
        self.x = x
        self.y = y
        self.speed = speed
        self.theta = theta
        self.id = id
        self.sample_time = sample_time
        self.path = np.array([x, y])
        self.ref_path = np.array([0, 0])
        
        # 停止标志：
        self.stop_flag_UGV=0


        self.start_time = start_time

        self.UGV_theta_acc_bound = bound[0]
        self.UGV_v_acc_bound = bound[1]
        self.UGV_v_bound = bound[2]

        # 自带武器优势参数
        self.weapn_effect = 0.1

        # 内置PID
        self.PID_theta = PID(k_pid[0], k_pid[1], k_pid[2])

        # 当前追踪目标
        self.ugv_aim_Tar = []

    def update_ugv_position(self, Target, matches):
        # 更新位置
        self.x += self.speed * math.cos(self.theta) * self.sample_time
        self.y += self.speed * math.sin(self.theta) * self.sample_time
        self.path = np.vstack((self.path, np.array([self.x, self.y])))

        

    def update_speed(self, v_acceleration, theta_acceleration):
        # 更新速度
        self.speed += v_acceleration * self.sample_time
        self.theta += theta_acceleration * self.sample_time

        self.speed =  np.clip(self.speed, -self.UGV_v_bound, self.UGV_v_bound)

class UGV_Group(UGV):
    def __init__(self, group_id, ugvs_params=None, ugv_list=None, empty_group=False):
        self.group_id = group_id
        if ugvs_params is not None:
            self.ugvs = [UGV(*params) for params in ugvs_params]
        elif ugv_list is not None:
            self.ugvs = ugv_list
        elif empty_group is True:
            self.ugvs = []

        self.stop_flag_Group=0

        self.ugv_phi = np.zeros((len(self.ugvs),2))
        self.ugv_pos = np.zeros((len(self.ugvs),2))

        self.aim_Tar:Hunting_Tar = None

        self.matches = np.zeros((len(self.ugvs), 2))
        self.D_Target_x = np.zeros((len(self.ugvs), len(self.ugvs)))
        self.D_Target_y = np.zeros((len(self.ugvs), len(self.ugvs)))
        self.D2Target = np.zeros((len(self.ugvs), len(self.ugvs)))

        self.D_2_hunt_pts = np.zeros((len(self.ugvs),2))

        self.pursuer_merge_x = np.zeros(len(self.ugvs))
        self.pursuer_merge_y = np.zeros(len(self.ugvs))
        self.pursuer_hunt_pts_x = np.zeros(len(self.ugvs))
        self.pursuer_hunt_pts_y = np.zeros(len(self.ugvs)) 
        self.pursuer_forces_x = np.zeros(len(self.ugvs))
        self.pursuer_forces_y = np.zeros(len(self.ugvs))
        self.target_forces_x = np.zeros(len(self.ugvs))
        self.target_forces_y = np.zeros(len(self.ugvs))

    def stop_group(self):
        [setattr(ugv, 'speed', 0) for ugv in self.ugvs]

    def update_aim_target(self, Target):
        self.aim_Tar = Target
        for ugv in self.ugvs:
            ugv.ugv_aim_Tar.append(Target.Tar_id)

    def update_group_positions(self):
        # 根据速度更新组内所有UGV的位置
        for ugv in self.ugvs:
            ugv.update_ugv_position(self.aim_Tar, self.matches)

        # 更新当前集群位置
        ugv_pos, ugv_phi, ugv_pos_id  = self.get_group_info()

        return ugv_pos, ugv_phi, ugv_pos_id
    
    def update_group_speeds(self, accelerations, theta_acc):
        # 更新组内所有UGV的速度
        for index, ugv in enumerate(self.ugvs):
            ugv.update_speed(accelerations[0][index], theta_acc[0][index])

    def get_group_info(self):
        ugv_pos = np.zeros((len(self.ugvs), 2))  # 使用 numpy.zeros 初始化矩阵
        ugv_phi = np.zeros((len(self.ugvs), 2))  # 使用 numpy.zeros 初始化矩阵

        ugv_pos_id = {}
        
        for index, ugv in enumerate(self.ugvs):
            ugv_pos[index] = np.array([ugv.x, ugv.y])
            ugv_phi[index] = np.array([ugv.theta, 0])  # 假设 phi 也是一个包含两个元素的向量

            ugv_pos_id[ugv.id] = np.array([ugv.x, ugv.y])
            

        self.ugv_phi = ugv_phi
        self.ugv_pos = ugv_pos

        self.D_inner_x, self.D_inner_y = get_Inner_Dis(self.ugv_pos)

        return ugv_pos, ugv_phi, ugv_pos_id
    
    def get_ugv_by_id(self, ugv_id):
        for ugv in self.ugvs:
            if ugv.id == ugv_id:
                return ugv
    
    def acceleration(self, all_ugv_force_id, flag_UGV):
        theta_acceleration = np.zeros((1,len(self.ugvs)))
        v_acceleration = np.zeros((1,len(self.ugvs)))

        # 一、比例导引率+固定速度
        if flag_UGV == 1:
            for index, ugv in enumerate(self.ugvs):
                
                # 角度
                tehta_d = np.arctan2(self.aim_Tar.hunt_ps[self.matches[index, 1]][1] - ugv.y, self.aim_Tar.hunt_ps[self.matches[index, 1]][0] - ugv.x)
                theta_acceleration[0][index] =  1.5 * calculate_angle_error(tehta_d, ugv.theta)
                theta_acceleration[0][index] = np.clip(theta_acceleration[0][index], -ugv.UGV_theta_acc_bound, ugv.UGV_theta_acc_bound)

                # 速度
                v_acceleration[0][index] = 0
                v_acceleration[0][index] = np.clip(v_acceleration[0][index], -ugv.UGV_v_acc_bound, ugv.UGV_v_acc_bound)
            
        # 二、人工势场提供全部信息
        if flag_UGV == 2:
            for index, ugv in enumerate(self.ugvs):
                
                # 角度
                self.pursuer_merge_x += self.pursuer_hunt_pts_x
                self.pursuer_merge_y += self.pursuer_hunt_pts_y
                # 根据人工势场计算出的偏转角度
                theta_d = np.arctan2(self.pursuer_merge_y[index], self.pursuer_merge_x[index])
                
                theta_acceleration[0][index] =  ugv.PID_theta.update_theta(theta_d, ugv.theta)
                theta_acceleration[0][index] = np.clip(theta_acceleration[0][index], -ugv.UGV_theta_acc_bound, ugv.UGV_theta_acc_bound)

                # 速度
                v_acceleration[0][index] = np.hypot(self.pursuer_merge_y[index], self.pursuer_merge_x[index]) - ugv.speed
                v_acceleration[0][index] = np.clip(v_acceleration[0][index], -ugv.UGV_v_acc_bound, ugv.UGV_v_acc_bound)
    
        # 二、人工势场做局部规划器， 全局比例导引率
        if flag_UGV == 3:
            for index, ugv in enumerate(self.ugvs):
                
                # 角度

                # 根据路径跟踪计算出的角度 angle_d
                # theta_d = np.arctan2(ugv.ref_path[-1][1] - ugv.y, ugv.ref_path[-1][0]  - ugv.x)
                
                # 直接跟随目标点
                theta_d = np.arctan2(self.aim_Tar.hunt_ps[self.matches[index, 1]][1] - ugv.y, self.aim_Tar.hunt_ps[self.matches[index, 1]][0] - ugv.x)
                
                # 根据人工势场计算出的偏转角度
                theta_d_APF = np.arctan2(self.pursuer_merge_y[index], self.pursuer_merge_x[index])
                
                if theta_d_APF != 0 :
                    theta_d = mid_angle(theta_d, theta_d_APF)
                    # theta_d = theta_d_APF
                else:
                    theta_d = theta_d

                theta_acceleration[0][index] =  ugv.PID_theta.update_theta(theta_d, ugv.theta)
                theta_acceleration[0][index] = np.clip(theta_acceleration[0][index], -ugv.UGV_theta_acc_bound, ugv.UGV_theta_acc_bound)

                # 速度
                v_acceleration[0][index] = 0
                v_acceleration[0][index] = np.clip(v_acceleration[0][index], -ugv.UGV_v_acc_bound, ugv.UGV_v_acc_bound)

        # 四、基于Vector2d的改进人工势场
        if flag_UGV == 4:
            for index, ugv in enumerate(self.ugvs):
                
        
                # 根据改进人工势场计算出的偏转角度
                theta_d = np.arctan2(all_ugv_force_id[ugv.id].deltaY, all_ugv_force_id[ugv.id].deltaX)
                
                theta_acceleration[0][index] =  ugv.PID_theta.update_theta(theta_d, ugv.theta)
                theta_acceleration[0][index] = np.clip(theta_acceleration[0][index], -ugv.UGV_theta_acc_bound, ugv.UGV_theta_acc_bound)

                # 速度
                # v_acceleration[0][index] = np.sqrt(self.pursuer_merge_y[index]**2 + self.pursuer_merge_x[index]**2)
                v_acceleration[0][index] = np.hypot(all_ugv_force_id[ugv.id].deltaY, all_ugv_force_id[ugv.id].deltaX) - ugv.speed
                v_acceleration[0][index] = np.clip(v_acceleration[0][index], -ugv.UGV_v_acc_bound, ugv.UGV_v_acc_bound)

        return v_acceleration, theta_acceleration
    
    def set_ref_path(self, all_group_match_id_dic, flag):
        # flag = 1：目标点轨迹即为参考轨迹
        if flag == 1:
            for index, ugv in enumerate(self.ugvs):
                ugv.ref_path =  self.aim_Tar.hunt_ps_store[all_group_match_id_dic[-1][ugv.id]]
    
    # def add_ugv(self, all_ugvs, ugv_id):
    # # 在外部UGV列表中查找指定ID的UGV
    #     for ugv in all_ugvs:
    #         if ugv.id == ugv_id:
    #             self.ugvs.append(ugv)
    #             self._update_internal_matrices()
    #             break

    def add_ugv(self, Ori_Group, ugv_id):
        new_ugv = Ori_Group.get_ugv_by_id(ugv_id)
        self.ugvs.append(new_ugv)

    def remove_ugv(self, ugv_id):
        self.ugvs = [ugv for ugv in self.ugvs if ugv.id != ugv_id]
        self._update_internal_matrices()

    def add_new_ugv(self, new_ugv:UGV):
    # 在外部UGV列表中查找指定ID的UGV
        self.ugvs.append(new_ugv)

    def _update_internal_matrices(self):
        num_ugvs = len(self.ugvs)
        self.ugv_phi = np.zeros((num_ugvs, 2))
        self.ugv_pos = np.zeros((num_ugvs, 2))
        self.matches = np.zeros((num_ugvs, 2))
        self.D_Target_x = np.zeros((num_ugvs, num_ugvs))
        self.D_Target_y = np.zeros((num_ugvs, num_ugvs))
        self.D2Target = np.zeros((num_ugvs, num_ugvs))
        self.D_2_hunt_pts = np.zeros((num_ugvs, 2))
        self.pursuer_merge_x = np.zeros(num_ugvs)
        self.pursuer_merge_y = np.zeros(num_ugvs)
        self.pursuer_hunt_pts_x = np.zeros(num_ugvs)
        self.pursuer_hunt_pts_y = np.zeros(num_ugvs)
        self.pursuer_forces_x = np.zeros(num_ugvs)
        self.pursuer_forces_y = np.zeros(num_ugvs)
        self.target_forces_x = np.zeros(num_ugvs)
        self.target_forces_y = np.zeros(num_ugvs)
    
class Hunting_Tar():
    def __init__(self, Tar_id, x, y, speed, theta, Tar_bound, r_hunt, start_time, Tar_value, dt):
        self.Tar_id = Tar_id
        self.x = x
        self.y = y
    
        self.speed = speed
        self.theta = theta
        self.sample_time = dt

        self.temp_pos = np.array([])
        self.aim_pos = np.array([])
        self.trace = np.array([x,y])
        # self.hunt_ps_store = [np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])]
        self.hunt_ps_store: Dict[int, List[List[float]]] = {}
        
        self.hunt_num = 3

        self.pur_group:UGV_Group = None

        self.Tar_theta_acc_bound = Tar_bound[0]
        self.Tar_v_acc_bound = Tar_bound[1]
        self.Tar_v_bound = Tar_bound[2]

        self.D2Pur = np.zeros((self.hunt_num, self.hunt_num))

        self.hunt_ps = np.zeros((1,self.hunt_num))

        self.escapee_force_x = np.zeros(self.hunt_num)
        self.escapee_force_y = np.zeros(self.hunt_num)
       
        self.r_hunt = r_hunt

        self.start_time = start_time

        self.Tar_value = Tar_value
        
        # 停止标志
        self.stop_flag_Tar=0

    def stop_tar(self):
        setattr(self, 'speed', 0)
        
    def update_pur_group(self, Group):
        self.pur_group = Group

    def update_Tar_position(self):
        
        # 更新位置
        self.x += self.speed * math.cos(self.theta) * self.sample_time
        self.y += self.speed * math.sin(self.theta) * self.sample_time
    
        self.aim_pos = np.array([self.x, self.y])

        self.trace = np.vstack((self.trace, np.array([self.x, self.y])))

        return self.aim_pos  
    
    ############################################# 速度与角度更新 ###########################################################


    def update_speed(self, v_acceleration, theta_acceleration):

        # 更新速度
        if self.stop_flag_Tar != 0:
            self.speed += v_acceleration * self.sample_time
            
            self.theta += theta_acceleration * self.sample_time

            self.speed =  np.clip(self.speed, -self.Tar_v_bound, self.Tar_v_bound)
        else:
            self.theta = 0
            self.speed = 0


        
    def acceleration(self, all_tar_force_id, flag_Tar):
        theta_acceleration = 0
        v_acceleration = 0

        # 一、人工势场_version_01 Target
        if flag_Tar == 1:
            theta_d = np.arctan2(self.escapee_force_y.sum(), self.escapee_force_x.sum())
        
            # theta_acceleration = theta_d - self.theta 
            theta_acceleration = calculate_angle_error(theta_d, self.theta)

            v_acc_x = self.escapee_force_x.sum()
            v_acc_y = self.escapee_force_y.sum()

            v_acceleration = np.sqrt(v_acc_x**2 + v_acc_y**2)

            v_acceleration =  np.clip(v_acceleration, -self.Tar_v_acc_bound, self.Tar_v_acc_bound)
            theta_acceleration =  np.clip(theta_acceleration, -self.Tar_theta_acc_bound, -self.Tar_theta_acc_bound)
        
        # 二、人工势场_version_02 Target
        # 和version_02的区别主要v_acceleration上，减去了当前速度
        if flag_Tar == 2:
            theta_d = np.arctan2(self.escapee_force_y.sum(), self.escapee_force_x.sum())
            # print(theta_d/np.pi)
            # theta_acceleration = theta_d - self.theta 
            theta_acceleration = 0.5 * calculate_angle_error(theta_d, self.theta)

            v_acc_x = self.escapee_force_x.sum()
            v_acc_y = self.escapee_force_y.sum()
            # print(np.sqrt(v_acc_x**2 + v_acc_y**2))
            v_acceleration = np.sqrt(v_acc_x**2 + v_acc_y**2) - self.speed

            v_acceleration =  np.clip(v_acceleration, -self.Tar_v_acc_bound, self.Tar_v_acc_bound)
            theta_acceleration =  np.clip(theta_acceleration, -self.Tar_theta_acc_bound, self.Tar_theta_acc_bound)
            
        # 三、人工势场_version_03 Target
        # 同时受到所有UGV的影响
        if flag_Tar == 3:
  
            theta_d = np.arctan2(all_tar_force_id[self.Tar_id].deltaY, all_tar_force_id[self.Tar_id].deltaX)

            theta_acceleration = 0.5 * calculate_angle_error(theta_d, self.theta)

            v_acceleration = np.hypot(all_tar_force_id[self.Tar_id].deltaY, all_tar_force_id[self.Tar_id].deltaX) - self.speed

            v_acceleration =  np.clip(v_acceleration, -self.Tar_v_acc_bound, self.Tar_v_acc_bound)
            theta_acceleration =  np.clip(theta_acceleration, -self.Tar_theta_acc_bound, self.Tar_theta_acc_bound)
        return v_acceleration, theta_acceleration
    
    def update_target_speeds(self, v_acc, theta_acc):
        self.update_speed(v_acc, theta_acc)

    ##################################################################################
    
    def hunt_points(self, flag):
        ugv_pos = self.pur_group.ugv_pos
        self.hunt_num = len(ugv_pos)
        r_hunt = self.r_hunt

        # 一、用0 120 240生成
        if flag == 1:
            
            for i in range(self.hunt_num):
                if i not in self.hunt_ps_store:
                    self.hunt_ps_store[i] = []
                self.hunt_ps_store[i].append([r_hunt * np.cos(0 + 2 * i * np.pi / self.hunt_num) + self.aim_pos[0], r_hunt * np.sin(0 + 2 * i * np.pi / self.hunt_num) + self.aim_pos[1]])
            
            # if self.hunt_num < len(self.hunt_ps_store):
            #     for i in range(self.hunt_num, len(self.hunt_ps_store)):
            #         self.hunt_ps_store[i].append([0, 0])

        # 二、用基于距离的优化生成
        # 初始角度猜测（等分点）
        if flag == 2:
            initial_angles = np.linspace(0, 2 * np.pi, self.hunt_num, endpoint=False)
            # 优化函数与优化结果
            constraints = {'type': 'eq', 'fun': angle_constraint}
            optimal_angles = (minimize(total_distance, initial_angles, args=(ugv_pos, (self.aim_pos[0], self.aim_pos[1]), self.r_hunt), bounds=[(0, 2*np.pi)]*self.hunt_num, constraints=constraints)).x
            self.hunt_ps = np.array([point_on_circle((self.aim_pos[0], self.aim_pos[1]), self.r_hunt, angle) for angle in optimal_angles]).tolist()
            
            for i in range(self.hunt_num):
                if i not in self.hunt_ps_store:
                    self.hunt_ps_store[i] = []
                self.hunt_ps_store[i].append(self.hunt_ps[i])

        # 三、角度、距离均考虑
        if flag ==3 :
            optimal_angles = vehicle_deployment_optimization(ugv_pos, self.aim_pos, r_hunt, k_d=0.9, k_theta=0.1)
            self.hunt_ps = np.array([point_on_circle((self.aim_pos[0], self.aim_pos[1]), self.r_hunt, angle) for angle in optimal_angles]).tolist()
            
            for i in range(self.hunt_num):
                if i not in self.hunt_ps_store:
                    self.hunt_ps_store[i] = []
                self.hunt_ps_store[i].append(self.hunt_ps[i])

        # 四、合同网协议角度、速度优化
        if flag == 4:
            optimal_angles = contract_net_angle_assign(ugv_pos, self.aim_pos, r_hunt, k_d=0.9, k_theta=0.1)
            self.hunt_ps = np.array([point_on_circle((self.aim_pos[0], self.aim_pos[1]), self.r_hunt, angle) for angle in optimal_angles]).tolist()
            
            for i in range(self.hunt_num):
                if i not in self.hunt_ps_store:
                    self.hunt_ps_store[i] = []
                self.hunt_ps_store[i].append(self.hunt_ps[i])

        # 五、做了更新优化后的考虑速度、角度的表达式
    

    def _calcu_and_match_hunt_pts(self, flag):
        ugv_pos = self.pur_group.ugv_pos

        D_Target_x = np.zeros((1, len(self.pur_group.ugvs)))
        D_Target_y = np.zeros((1, len(self.pur_group.ugvs)))

        D2Target = np.zeros((len(self.pur_group.ugvs), len(self.pur_group.ugvs)))

        D_2_hunt_pts_x = np.zeros((len(self.pur_group.ugvs), len(self.pur_group.ugvs)))
        D_2_hunt_pts_y = np.zeros((len(self.pur_group.ugvs), len(self.pur_group.ugvs)))
        
        # flag =1：基于距离的匈牙利分配算法
        # 这样写能够自适应到个数吗？
        if flag == 1:
            for i in range(len(ugv_pos)):
                D_Target_x[0][i] = ugv_pos[i][0] - self.x
            
                D_Target_y[0][i] = ugv_pos[i][1] - self.y

                for j in range(len(ugv_pos)):
        
                    D2Target[i, j] = np.hypot(ugv_pos[i][0] - self.hunt_ps_store[j][-1][0], ugv_pos[i][1] - self.hunt_ps_store[j][-1][1])
                    
                    D_2_hunt_pts_x[i][j] = ugv_pos[i][0] - self.hunt_ps_store[j][-1][0]
                    D_2_hunt_pts_y[i][j] = ugv_pos[i][1] - self.hunt_ps_store[j][-1][1]

            matches, D_2_hunt_pts = match(D2Target, D_2_hunt_pts_x, D_2_hunt_pts_y)

            # 替换为id
            matches[:,0] = np.array([ugv.id for ugv in self.pur_group.ugvs])
            matches = matches.astype(int)
            
            self.D2Pur = - D2Target

        return matches, D_Target_x, D_Target_y, D2Target, D_2_hunt_pts

class Obstacles:
    """
    障碍物类，接受多个点作为障碍物位置
    """
    def __init__(self, *points):
        self.positions = [np.array([x, y]) for x, y in points]
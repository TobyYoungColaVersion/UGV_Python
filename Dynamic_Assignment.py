from typing import Dict, List
import math
import random
import numpy as np
import copy
from scipy.optimize import minimize, linear_sum_assignment, linprog
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import heapq

from Other_Func import calculate_angle_error, point_on_circle, total_distance, angle_constraint, mid_angle, apollonius_circle, match, get_Inner_Dis, PID
from Artifical_Potential_Field import APF_original, APF_Improved
from Sim_Class import UGV, UGV_Group, Hunting_Tar, Obstacles         
from CBBA_Allocation import cbba    
from Auction_Allocation import auction_algorithm, contract_net_protocol

#################################### 动态分配函数 #################################
class Dynamic_Assignment():
    def __init__(self, init_Params):
        self.all_Pairs = {}
        self.Pairs_init(init_Params['Pairs'])

        # 存储全部UGV的字典
        self.all_ugvs_list:list[UGV] = {}
        self.all_ugvs_dict:Dict[int, UGV] = {}
        
        # 全部UGV当前位置
        self.all_ugv_pos_id_dic = {}
        self.all_ugv_pos_id_list = []

        # 全部目标当前位置
        self.all_tar_pos_id_dic = {}    
        self.all_tar_pos_id_list = []

        # 全部UGV当前受力
        self.all_ugv_force_id = {}

        # 全部Target当前受力
        self.all_tar_force_id = {}

        # 全部匹配
        self.all_group_match_id_dic = []

        
        # 态势指标
        # 态势指标总存储 ：self.situation
        self.situation:Dict = {}

        # 指标0：每个点是否被围   
        self.is_hunted_dic = {}

        # 指标1：每个目标点维护一个效用值 字典{目标点：每个个体的}
        self.all_ugv_utility_value:Dict[Hunting_Tar, Dict[UGV, float]] = {}

        self.iter = 0

        # 每一步指派结果的存储
        self.Groups_UGV_Assign_Dict:Dict[UGV_Group, List[UGV]] = {}

        self.Assign_lists_store=[]

        self.store = {}

        # 其他参数
        self.hunted_threshold = init_Params['Other_Params']['hunted_threshold']

          # 效用函数参数

        self.Max_Path = init_Params['Other_Params']['Max_Path']
        self.R_dens = init_Params['Other_Params']['R_dens']
        self.k_dens = init_Params['Other_Params']['Tar_Benefit_Params']['k_dens']
        self.k_val = init_Params['Other_Params']['Tar_Benefit_Params']['k_val'] 

        self.k_dis = init_Params['Other_Params']['UGVs_Cost_Params']['k_dis'] 
        self.k_theta = init_Params['Other_Params']['UGVs_Cost_Params']['k_theta'] 
        self.k_weapon = init_Params['Other_Params']['UGVs_Cost_Params']['k_weapon'] 

        # 障碍物
        self.obstacle = init_Params['Other_Params']['Obstacle']
########### 基础操作 ############

    def Pairs_init(self, init_Pairs:Dict[Hunting_Tar, UGV_Group]):
        
        self.all_Pairs = init_Pairs
        for Target_cur in init_Pairs:
            init_Pairs[Target_cur].update_aim_target(Target_cur)

    def get_pairs(self):
        return self.all_Pairs   
    
    def get_all_ugvs(self):
        ugv_temp_list:List[UGV] = []

        for index, Target_cur in enumerate(self.all_Pairs):
                ugv_temp_list += self.all_Pairs[Target_cur].ugvs
        
        for ugv in ugv_temp_list:

            self.all_ugvs_dict[ugv.id] = ugv

        self.all_ugvs_list = ugv_temp_list

        return self.all_ugvs_dict, self.all_ugvs_list
    
    def get_ugvs_list_by_id(self, id_list):
        ugvs_list:list[UGV] = []
        for id in id_list:
            ugvs_list.append(self.all_ugvs_dict[id])
        return ugvs_list
    
    def change_group(self):
        pass

########### 基础操作 ############

########### 态势判断 ############

    def _get_situation_awareness(self):
        
         # self.situation['Targets_Benefit'] = {Tar_0: Benefit_0, Tar_1: Benefit_1}
        self.situation['Targets_Benefit'] = self._get_Targets_Benefit()

        self.situation['UGVs_Cost'] = self._get_UGVs_Cost(self.all_ugvs_dict)

        # 最终：根据前面的全部内容，计算当前所有UGV对每个目标的效用值
        # 格式：{Target_0 : { ugv.id : utility_value}, Target_1 : { ugv.id : utility_value}}
        all_ugv_utility_value = self._get_Utility_value()


    def solve_dynamic_assign(self, flag, iter):

        # 根据态势信息解全部优化问题

        # 0、根据态势信息获取全局效用值
        all_ugv_utility_value = self.all_ugv_utility_value

        # 1、根据全局效用值进行分配列表计算
        # Assign_lists = self._calculate_Assign_Dict_by_situation(all_ugv_utility_value)

        # 1、根据态势判断来计算
        Assign_lists = self._allocate_ugvs_to_targets(flag)
        
        self.Assign_lists_store.append(Assign_lists)

        # 2、返回分配列表
        for i in range(len(Assign_lists)):
            self.Groups_UGV_Assign_Dict[i] = self.get_ugvs_list_by_id(Assign_lists[i])
        
        return self.Groups_UGV_Assign_Dict
    
    def _allocate_ugvs_to_targets(self, flag):

        UGVs_Cost = self.situation['UGVs_Cost']
        targets_benefit = self.situation['Targets_Benefit']

        ugv_allocation = defaultdict(list)
        assigned_ugvs = set()

      # if flag=1 ：最简单的匈牙利分配方法 
    # 首先为每个目标分配至少3个UGV
        # for Target_cur in self.all_Pairs:
        #     min_heap = [(cost, ugv_id) for ugv_id, cost in UGVs_Cost[Target_cur].items() if ugv_id not in assigned_ugvs]
        #     heapq.heapify(min_heap)

        #     while len(ugv_allocation[Target_cur]) < 3 and min_heap:
        #         cost, ugv_id = heapq.heappop(min_heap)
        #         ugv_allocation[Target_cur].append(self.all_ugvs_dict[ugv_id])
        #         assigned_ugvs.add(ugv_id)

        # # 然后为每个目标分配最多5个UGV
        # for Target_cur in self.all_Pairs:
        #     min_heap = [(cost, ugv_id) for ugv_id, cost in UGVs_Cost[Target_cur].items() if ugv_id not in assigned_ugvs]
        #     heapq.heapify(min_heap)

        #     while len(ugv_allocation[Target_cur]) < 5 and min_heap:
        #         cost, ugv_id = heapq.heappop(min_heap)
        #         ugv_allocation[Target_cur].append(self.all_ugvs_dict[ugv_id])
        #         assigned_ugvs.add(ugv_id)

        # # 转换为包含 UGV ID 的列表列表
        # ordered_allocation = [[ugv.id for ugv in ugv_allocation[Target_cur]] for Target_cur in self.all_Pairs]

        #  if flag = 2 :ECBBA算法
        # ordered_allocation = cbba(UGVs_Cost, max_iterations=10, min_ugvs_per_target=3, max_ugvs_per_target=5)
        
        # if flag = 3: 贪心算法
        # ordered_allocation = auction_algorithm(UGVs_Cost)

        # if flag = 4: 合同网协议
        ordered_allocation = contract_net_protocol(UGVs_Cost)
        return ordered_allocation


    def _calculate_Assign_Dict_by_situation(self, all_ugv_utility_value, min_ugvs_per_target=3, max_ugvs_per_target=5):
            # Create a list of all targets and UGVs
        all_targets = list(all_ugv_utility_value.keys())
        all_ugvs = list({ugv for target in all_ugv_utility_value for ugv in all_ugv_utility_value[target]})

        # Count the number of targets and UGVs
        num_targets = len(all_targets)
        num_ugvs = len(all_ugvs)

        # Initialize lists to store coefficients and constraints
        c = []
        A_ub = []
        b_ub = []

        # Construct the objective function coefficients vector c
        for target in all_targets:
            for ugv in all_ugvs:
                c.append(-all_ugv_utility_value[target].get(ugv, 0))

        # Construct the inequality constraints matrices A_ub and b_ub
        # Constraint 1: Each UGV can be assigned to at most one target
        for ugv in all_ugvs:
            constraint_row = [0] * (num_targets * num_ugvs)
            for idx, target in enumerate(all_targets):
                ugv_idx = all_ugvs.index(ugv)
                constraint_row[idx * num_ugvs + ugv_idx] = 1
            A_ub.append(constraint_row)
            b_ub.append(1)

        # Constraint 2: Each target must be assigned between min_ugvs_per_target and max_ugvs_per_target UGVs
        for target in all_targets:
            constraint_row = [0] * (num_targets * num_ugvs)
            target_idx = all_targets.index(target)
            for ugv in all_ugvs:
                ugv_idx = all_ugvs.index(ugv)
                constraint_row[target_idx * num_ugvs + ugv_idx] = 1
            A_ub.append(constraint_row.copy())
            b_ub.append(max_ugvs_per_target)
            
            constraint_row = [0] * (num_targets * num_ugvs)
            for ugv in all_ugvs:
                ugv_idx = all_ugvs.index(ugv)
                constraint_row[target_idx * num_ugvs + ugv_idx] = -1
            A_ub.append(constraint_row.copy())
            b_ub.append(-min_ugvs_per_target)

        # Solve the linear programming problem using linprog
        bounds = [(0, 1)] * (num_targets * num_ugvs)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        # Extract and format the result
        Assign_lists = []
        for idx, target in enumerate(all_targets):
            target_assignments = []
            for ugv_idx, ugv in enumerate(all_ugvs):
                if res.x[idx * num_ugvs + ugv_idx] > 0.5:  # Consider assignment if x > 0.5 (rounding to binary)
                    target_assignments.append(ugv)
            Assign_lists.append(target_assignments)

        return Assign_lists
   

    # 目标是否被围捕
    def is_hunted(self):
        # 判断当前是否完成围捕 --- 其实就是判断是否到达势点
        is_hunted_dic = {}

        for Target_cur in self.all_Pairs:
            Group_cur = self.all_Pairs[Target_cur]
            sum_distances_hunt = sum(np.linalg.norm(Group_cur.D_2_hunt_pts, axis=1))
            
            # 距离和小于hunted_threshold则为围捕成功
            is_hunted_dic[Target_cur.Tar_id] = 1 if sum_distances_hunt < self.hunted_threshold else 0      
        
        self.situation['is_hunted_dic'] = is_hunted_dic
        
        return is_hunted_dic
    
########### 态势判断 ############


########### 态势计算 ############

    def _get_Targets_Benefit(self):
     
        self.situation['Targets_Benefit']= {}

        target_density = self._get_Tar_dens()

        for index, Target_cur in enumerate(self.all_Pairs):

            dens_cur = target_density[Target_cur]

            self.situation['Targets_Benefit'][Target_cur] = self.k_dens * dens_cur + self.k_val * Target_cur.Tar_value
    
        return self.situation['Targets_Benefit']
    
    def _get_Tar_dens(self):
        R_dens = self.R_dens
        target_density = {}

        for Target_cur in self.all_Pairs:
            count = 0
            for Target in self.all_Pairs:
                distance = np.hypot(Target_cur.x - Target.x, Target_cur.y - Target.y)
                if distance <= R_dens and distance !=0 :
                    count += 1
            target_density[Target_cur] = count

        return target_density
    
    def _get_UGVs_Cost(self, all_ugvs_dict:Dict[int, UGV]):

        k_dis = self.k_dis
        k_theta = self.k_theta
        k_weapon = self.k_weapon

        distances_cost = {}
        theta_cost = {}

        # 这一项需要做减法
        comprehensive_cost = {}

        overall_cost = {}

        for Target in self.all_Pairs:
            Target_distance = {}
            Target_theta_relative = {}
            UGV_weapon_benefit = {}\
            
            overall_Target_cost = {}

            for ugv_id, ugv in all_ugvs_dict.items():
                
                # 相对目标的距离
                distance = np.hypot(Target.x - ugv.x, Target.y - ugv.y)

                # 指数形式 
                # distance = math.exp(-np.hypot(Target.x - ugv.x, Target.y - ugv.y)**2)
                Target_distance[ugv_id] = k_dis * distance

                relative_theta = np.abs(calculate_angle_error(np.arctan2(Target.y - ugv.y, Target.x - ugv.x), ugv.theta)/np.pi)
                Target_theta_relative[ugv_id] = k_theta * relative_theta

                weapon_effect = ugv.weapn_effect
                UGV_weapon_benefit[ugv_id] = k_weapon * weapon_effect

                # 单个目标总cost计算
                overall_Target_cost[ugv_id] = k_dis * distance + k_theta * relative_theta - k_weapon * weapon_effect


            distances_cost[Target] = Target_distance
            theta_cost[Target] = Target_theta_relative
            comprehensive_cost[Target] = UGV_weapon_benefit
            
            # 总cost计算
            overall_cost[Target] = overall_Target_cost

        return overall_cost
    
    def _get_Utility_value(self):

        # 根据当前态势计算效用值
        temp_dict_list = [{} for tar in self.all_Pairs]

        for ugv_id in self.all_ugvs_dict:

            for i in range(len(temp_dict_list)):
            
                temp_dict_list[i][ugv_id] = ugv_id + 0.5

        for index, Target_cur in enumerate(self.all_Pairs):
            
            self.all_ugv_utility_value[Target_cur] = temp_dict_list[index]

        # 格式:{Target_0 : { ugv.id : utility_value}, Target_1 : { ugv.id : utility_value}}

        return self.all_ugv_utility_value


########### 态势计算 ############


########### 各类更新 ############
    def update_all_position(self):
        
        temp_ugv_pos_id_dic = [0 for _ in range(len(self.all_Pairs))]
        self.all_tar_pos_id_dic = {}
        self.all_ugv_pos_id_dic = {}

        for index, Target_cur in enumerate(self.all_Pairs):
            self.all_tar_pos_id_dic[Target_cur.Tar_id] = Target_cur.update_Tar_position()
            _, _, temp_ugv_pos_id_dic[index] = self.all_Pairs[Target_cur].update_group_positions()

        self.all_ugv_pos_id_dic = {k: v for d in temp_ugv_pos_id_dic for k, v in d.items()}

        self.all_ugv_pos_id_list = [self.all_ugv_pos_id_dic[key] for key in sorted(self.all_ugv_pos_id_dic.keys())]

        self.all_tar_pos_id_list = [self.all_tar_pos_id_dic[key] for key in sorted(self.all_tar_pos_id_dic.keys())]

        return self.all_ugv_pos_id_dic, self.all_tar_pos_id_dic
    
    def update_all_hunt_pts(self, flag):

        for index, Target_cur in enumerate(self.all_Pairs):
            Target_cur.hunt_points(flag)
        
    
    def update_all_ref_path(self, flag):
        for index, Target_cur in enumerate(self.all_Pairs):
            self.all_Pairs[Target_cur].set_ref_path(self.all_group_match_id_dic, flag)
        return 0
    
    def update_all_matches(self, flag):
        temp_group_match_id_list = [0 for _ in range(len(self.all_Pairs))]

        for index, Target_cur in enumerate(self.all_Pairs):
            self.all_Pairs[Target_cur].matches, self.all_Pairs[Target_cur].D_Target_x, self.all_Pairs[Target_cur].D_Target_y, self.all_Pairs[Target_cur].D2Target, self.all_Pairs[Target_cur].D_2_hunt_pts = Target_cur._calcu_and_match_hunt_pts(flag)
            temp_group_match_id_list[index] = self.all_Pairs[Target_cur].matches

        all_group_match_id_ndarray = np.vstack(temp_group_match_id_list)
        self.all_group_match_id_dic.append({row[0]: row[1] for row in all_group_match_id_ndarray}) 

        return self.all_group_match_id_dic
    
    def update_all_APF(self, APF_param, flag):
        
        R_EP, eta_inner, R_sensor, eta_Tar, R_PP, eta_hunt_pts = APF_param    

        
        if flag == 1:

            # flag = 1 ：改进的人工势场
            for index, Target_cur in enumerate(self.all_Pairs):
                # apf_now = APF_original(self.all_Pairs[Target_cur], Target_cur)
                # self.all_Pairs[Target_cur].pursuer_merge_x, self.all_Pairs[Target_cur].pursuer_merge_y, self.all_Pairs[Target_cur].pursuer_hunt_pts_x, self.all_Pairs[Target_cur].pursuer_hunt_pts_y, self.all_Pairs[Target_cur].pursuer_forces_x, self.all_Pairs[Target_cur].pursuer_forces_y, self.all_Pairs[Target_cur].target_forces_x, self.all_Pairs[Target_cur].target_forces_y, Target_cur.escapee_force_x, Target_cur.escapee_force_y = apf_now.calculate_forces(R_EP, eta_inner, R_sensor, eta_Tar, R_PP, eta_hunt_pts)
                
                # 用vec形式的APF分别计算Tars和ugvs的受力
                start = (Target_cur.x , Target_cur.y)
                goal = (0, 0)
                apf_tar = APF_Improved(start, goal, self.all_ugv_pos_id_list, k_att=0, k_rep=eta_Tar, rr=R_sensor, goal_threshold=30)
                self.all_tar_force_id[Target_cur.Tar_id] = apf_tar.all_force()

                for ugv in self.all_Pairs[Target_cur].ugvs:
                    start = (ugv.x, ugv.y)
                    # goal = tuple(self.all_Pairs[Target_cur].aim_Tar.hunt_ps[self.all_Pairs[Target_cur].matches[index%3, 1]])
                    goal = tuple(self.all_Pairs[Target_cur].aim_Tar.hunt_ps_store[self.all_group_match_id_dic[-1][ugv.id]][-1])
                    temp_dict = self.all_ugv_pos_id_dic.copy()
                    del temp_dict[ugv.id]
                    other_ugvs = [temp_dict[key] for key in sorted(temp_dict.keys())]

                    all_obs_and_ugvs = other_ugvs + self.obstacle.positions

                    # other_ugvs =  self.all_ugv_pos_id_list[:ugv.id] + self.all_ugv_pos_id_list[ugv.id+1:]
                    apf = APF_Improved(start, goal, all_obs_and_ugvs, k_att=eta_hunt_pts, k_rep=eta_inner, rr=R_EP, goal_threshold=30)
                    self.all_ugv_force_id[ugv.id] = apf.all_force()

            # flag = 2 ：狼群粒子算法

        return self.all_ugv_force_id, self.all_ugv_force_id
    
        

    def update_all_acceleration(self, flag_UGV, flag_Tar):
        for index, Target_cur in enumerate(self.all_Pairs):
            v_acc_UGVs, theta_acc_UGVs = self.all_Pairs[Target_cur].acceleration(self.all_ugv_force_id, flag_UGV)
            self.all_Pairs[Target_cur].update_group_speeds(v_acc_UGVs, theta_acc_UGVs)

            v_acc_Tar, theta_acc_Tar = Target_cur.acceleration(self.all_tar_force_id, flag_Tar)
            Target_cur.update_target_speeds(v_acc_Tar, theta_acc_Tar)

        return 0


###########################


################################## 根据态势判断的最终执行 #########################

    # 首先根据态势感知得到的新Targets初始化任务管理器的Pairs
    def all_pairs_re_init(self, Groups_list:List[UGV_Group], new_Targets_list:List[Hunting_Tar]):
        
        # 清空分配字典
        Groups_UGV_Assign_Dict:Dict[int, list[UGV]] = {}

         # 分配列表及配对初始化
        for index, Target_cur in enumerate(new_Targets_list):
            
            # 这里的命名还是再思考一下
            if index > len(Groups_list)-1:

                Group_cur:UGV_Group = UGV_Group(group_id=index, empty_group=True)
                
                Groups_list.append(Group_cur)
                self.all_Pairs[Target_cur] = Group_cur
                Target_cur.update_pur_group(Group_cur)
            
            else:
                Group_cur = self.all_Pairs[Target_cur]
                Target_cur.update_pur_group(Group_cur)

            # 分配字典初始化
            Groups_UGV_Assign_Dict[Group_cur.group_id] = []
        
        new_Groups_list = Groups_list
        self.Groups_UGV_Assign_Dict = Groups_UGV_Assign_Dict
    
        return Groups_UGV_Assign_Dict,  new_Groups_list

    def UGV_Assign(self, Groups_list:List[UGV_Group], Targets_list:List[Hunting_Tar], Groups_UGV_Assign_Dict, iter):
        
        # 根据 Groups_UGV_Assign_Dict 进行 List[UGV]分配

        Groups_UGV_Assign_Dict_cur = Groups_UGV_Assign_Dict
        
        # 将list赋给各Group
        for index, Target_cur in enumerate(self.all_Pairs):
            Group_cur:UGV_Group = self.all_Pairs[Target_cur]
            Group_cur.ugvs =  Groups_UGV_Assign_Dict_cur[Group_cur.group_id]
            Group_cur.update_aim_target(Target_cur)
            
            # # 围捕成功停止逻辑：围捕成功、对应的tar、ugvs全部暂停
            # if self.situation['is_hunted_dic'][Target_cur.Tar_id] == 1:
            #     Target_cur.stop_tar()
            #     self.all_Pairs[Target_cur].stop_group()
    


    # 最终需要执行的任务分配函数
    def Target_Assignment(self, Groups_list:List[UGV_Group], new_Targets_list:List[Hunting_Tar], iter:int, flag):
        self.iter  = iter

        all_ugvs_dict, _ = self.get_all_ugvs()

        # 初始化分配
        _, new_Groups_list = self.all_pairs_re_init(Groups_list, new_Targets_list)

        self._get_situation_awareness()
        
        # 形成从Group指向UGVs的分配模式

        if iter == 0 or iter == 25:
            self.store = self.solve_dynamic_assign(flag, iter)

        # Groups_UGV_Assign_Dict = self.solve_dynamic_assign(iter)

        self.UGV_Assign(new_Groups_list, new_Targets_list, self.store, iter)

        self.is_hunted()
        
        for index, Target_cur in enumerate(self.all_Pairs):

            # 围捕成功停止逻辑：围捕成功、对应的tar、ugvs全部暂停
            if self.situation['is_hunted_dic'][Target_cur.Tar_id] == 1:
                Target_cur.stop_tar()
                Target_cur.stop_flag_Tar=1
                # if Target_cur.stop_flag_Tar != 1:
                #     Target_cur.stop_flag_Tar=1
                self.all_Pairs[Target_cur].stop_group()
            # else:
            #     if Target_cur.stop_flag_Tar == 1:
            #         Target_cur.stop_tar()
            #         self.all_Pairs[Target_cur].stop_group()
        return new_Groups_list
    
    


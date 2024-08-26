"""
改进人工势场，解决不可达问题，仍存在局部最小点问题
"""
import matplotlib.pyplot as plt
import math
import random
import numpy as np

class Vector2d():
    """
    2维向量, 支持加减, 支持常量乘法(右乘)
    """
    def __init__(self, x, y):
        self.deltaX = x
        self.deltaY = y
        self.length = -1
        self.direction = [0, 0]
        self.vector2d_share()

    def vector2d_share(self):
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()):
            deltaX, deltaY = self.deltaX, self.deltaY
            self.deltaX = deltaY[0] - deltaX[0]
            self.deltaY = deltaY[1] - deltaX[1]
            self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length]
            else:
                self.direction = None
        else:
            self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length]
            else:
                self.direction = None

    def __add__(self, other):
        """
        + 重载
        :param other:
        :return:
        """
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.vector2d_share()
        return vec

    def __sub__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.vector2d_share()
        return vec

    def __mul__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.vector2d_share()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(self.deltaX, self.deltaY, self.length,
                                                                             self.direction)


class APF():
    """
    人工势场寻路
    """

    def __init__(self, start, goal, obstacles:np.ndarray, k_att: float, k_rep: float, rr: float,
                 goal_threshold: float):
        """
        :param start: 起点
        :param goal: 终点
        :param obstacles: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threshold: 离目标点小于此值即认为到达目标点
        :param is_plot: 是否绘图
        """
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.goal_threashold = goal_threshold
        
    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        att = (self.goal - self.current_pos) * self.k_att  # 方向由机器人指向目标点
        # att = (self.goal - self.current_pos) * 0  # 方向由机器人指向目标点
        return att

    def repulsion(self):
        """
        斥力计算
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            t_vec = self.current_pos - obstacle
            if (t_vec.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep += Vector2d(t_vec.direction[0], t_vec.direction[1]) * self.k_rep * (
                        1.0 / t_vec.length - 1.0 / self.rr) / (t_vec.length ** 2)  # 方向由障碍物指向机器人
        return rep


class APF_Improved(APF):
    def __init__(self, start, goal, obstacles:np.ndarray, k_att: float, k_rep: float, rr: float,
                 goal_threshold: float, is_goal=False):
        """
        :param start: 起点
        :param goal: 终点
        :param obstacles: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threshold: 离目标点小于此值即认为到达目标点
        :param is_plot: 是否绘图

        # 增加一下将传入参数变为元组/list的过程
        """
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        
        self.goal_threashold = goal_threshold

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            obs_to_rob = self.current_pos - obstacle
            rob_to_goal = self.goal - self.current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector2d(obs_to_rob.direction[0], obs_to_rob.direction[1]) * self.k_rep * (
                        1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector2d(rob_to_goal.direction[0], rob_to_goal.direction[1]) * self.k_rep * ((1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep +=(rep_1+rep_2)
        return rep
    
    def all_force(self):
        all_force = self.repulsion() + self.attractive()
        return all_force
    









########## 自己写的原始版本APF#####
# APF Function
class APF_original():
    def __init__(self, Group, Target):
        self.Group = Group
        self.Target = Target
    
    def apf_force(self, d_x, d_y, r, eta):
        distance = np.sqrt(d_x**2 + d_y**2)
        if distance < 1e-6:  # 防止除以零
            distance = 1e-6
        force_magnitude = eta * (1 / distance - 1 / r)/ distance**2 if distance < r else 0
        force_x = force_magnitude * (d_x / distance)
        force_y = force_magnitude * (d_y / distance)
        return force_x, force_y
    
    def apf_force_Gravity(self, d_x, d_y, r, eta):
        distance = np.sqrt(d_x**2 + d_y**2)
        if distance < 1e-6:  # 防止除以零
            distance = 1e-6
        force_magnitude = eta * distance if distance < r else 0
        force_x = force_magnitude * (d_x / distance)
        force_y = force_magnitude * (d_y / distance)
        return force_x, force_y

    def calculate_forces(self, R_EP, eta_inner, R_sensor, eta_Tar, R_PP, eta_hunt_pts):
        D_inner_x = self.Group.D_inner_x
        D_inner_y = self.Group.D_inner_y
        D_Target_x = self.Group.D_Target_x
        D_Target_y = self.Group.D_Target_y
        
        D_2_hunt_pts = self.Group.D_2_hunt_pts

        num_pursuers = D_inner_x.shape[0]
        
        # 追捕者之间的相互作用力
        pursuer_forces_x = np.zeros(num_pursuers)
        pursuer_forces_y = np.zeros(num_pursuers)
        for i in range(num_pursuers):
            for j in range(num_pursuers):
                if i != j:
                    force_x, force_y = self.apf_force(-D_inner_x[i][j], -D_inner_y[i][j], R_EP, eta_inner)
                    pursuer_forces_x[i] += force_x
                    pursuer_forces_y[i] += force_y

        # 追捕者受追捕点的引力
        pursuer_hunt_pts_x = np.zeros(num_pursuers)
        pursuer_hunt_pts_y = np.zeros(num_pursuers)
        for i in range(num_pursuers):
                force_x, force_y = self.apf_force_Gravity(-D_2_hunt_pts[i][0], -D_2_hunt_pts[i][1], R_PP, eta_hunt_pts)
                pursuer_hunt_pts_x[i] += force_x
                pursuer_hunt_pts_y[i] += force_y
            
        
        # 追捕者受到的逃跑者的斥力
        target_forces_x = np.zeros(num_pursuers)
        target_forces_y = np.zeros(num_pursuers)
        for i in range(num_pursuers):
            force_x, force_y = self.apf_force(D_Target_x[0][i], D_Target_y[0][i], R_EP, eta_inner)
            target_forces_x[i] += force_x
            target_forces_y[i] += force_y
 
        # 逃跑者受到的追捕者的斥力
        escapee_force_x = np.zeros(num_pursuers)
        escapee_force_y = np.zeros(num_pursuers)
        for i in range(num_pursuers):
            force_x, force_y = self.apf_force(-D_Target_x[0][i], -D_Target_y[0][i], R_sensor, eta_Tar)
            escapee_force_x[i] += force_x
            escapee_force_y[i] += force_y

        # 追捕者所受合力
        pursuer_merge_x = pursuer_forces_x + target_forces_x
        pursuer_merge_y = pursuer_forces_y + target_forces_y

        
        # print("追捕者集群内相互作用的合力 (x方向):", pursuer_forces_x)
        # print("追捕者集群内相互作用的合力 (y方向):", pursuer_forces_y)
        # print("追捕者分别受到逃跑者的斥力 (x方向):", target_forces_x)
        # print("追捕者分别受到逃跑者的斥力 (y方向):", target_forces_y)
        # print("追捕者分别受到合力 (x方向):", pursuer_merge_x)
        # print("追捕者分别受到合力 (y方向):", pursuer_merge_y)
        # print("逃跑者受追捕者的斥力 (x方向):", escapee_force_x)
        # print("逃跑者受追捕者的斥力 (y方向):", escapee_force_y)
        return pursuer_merge_x, pursuer_merge_y, pursuer_hunt_pts_x, pursuer_hunt_pts_y, pursuer_forces_x, pursuer_forces_y, target_forces_x, target_forces_y, escapee_force_x, escapee_force_y
 
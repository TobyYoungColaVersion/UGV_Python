import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize, linear_sum_assignment
import copy
import random

###################################################################################

def calculate_angle_error(aim_angle, current_angle):
    error = aim_angle - current_angle
    # Normalize error to be within [-π, π]
    error = (error + math.pi) % (2 * math.pi) - math.pi
    return error

# 计算单个点的坐标
def point_on_circle(center, radius, angle):
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return np.array([x, y])

# 目标函数，计算总距离
def total_distance(angles, captors, center, radius):
    points = [point_on_circle(center, radius, angle) for angle in angles]
    total_dist = 0
    for point in points:
        distances = np.linalg.norm(captors - point, axis=1)
        total_dist += np.min(distances)
    return total_dist

# 等式约束，确保角度差为120度
def angle_constraint(angles):
    N = len(angles)  # N 等分
    return [angles[i] - angles[i-1] - 2*np.pi/N for i in range(1, N)] + [angles[0] + 2*np.pi - angles[-1] - 2*np.pi/N]

def mid_angle(theta1, theta2):
    # Ensure the angles are between -pi and pi
    theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
    theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))

    # Compute the mid angle
    sin_avg = (np.sin(theta1) + np.sin(theta2)) / 2
    cos_avg = (np.cos(theta1) + np.cos(theta2)) / 2
    theta_mid = np.arctan2(sin_avg, cos_avg)

    return theta_mid


    # 绘制阿波罗尼奥斯圆
def apollonius_circle(pursuer, vp, evader, ve):

    px, py = pursuer
    ex, ey = evader
    
    lambda_0 = ve/vp if vp!=0 else 1.732/2
    
     # 计算两个点之间的距离
    d = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
    
    # 阿波罗尼奥斯圆的中心和半径
    r = d * lambda_0 / np.abs(lambda_0**2 - 1)

    d_OE = d * lambda_0**2 / np.abs(lambda_0**2 - 1)

    cx = ex + d_OE * np.cos(np.arctan2(py-ey, px-ex))
    cy = ey + d_OE * np.sin(np.arctan2(py-ey, px-ex))
    
    
    return cx, cy, r 


# 定义PID控制器类
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, setpoint, current_value):
        self.setpoint = setpoint
        error = self.setpoint - current_value

        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output
    
    def update_theta(self, setpoint, current_value):
        self.setpoint = setpoint
       
        error = calculate_angle_error(self.setpoint, current_value)
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

def match(D, D_2_hunt_pts_x, D_2_hunt_pts_y):
    D = np.array(D)
    # 匈牙利算法分配目标点
    row_ind, col_ind = linear_sum_assignment(D)
    matches = np.zeros((len(row_ind), 2), dtype=int)
    D_2_hunt_pts = np.zeros((len(row_ind),2))

    for i in range(len(row_ind)):
        matches[i, 0] = row_ind[i]
        matches[i, 1] = col_ind[i]

        D_2_hunt_pts[i][0] = D_2_hunt_pts_x[i][col_ind[i]]
        D_2_hunt_pts[i][1] = D_2_hunt_pts_y[i][col_ind[i]]

    return matches, D_2_hunt_pts

def get_Inner_Dis(ugv_pos):
    # 将列表转换为numpy数组
    ugv_pos = np.array(ugv_pos)

    # 获取点的数量
    num_points = len(ugv_pos)

    # 初始化D_x和D_y为全零矩阵
    D_x = np.zeros((num_points, num_points))
    D_y = np.zeros((num_points, num_points))

    # 计算所有点对之间的距离
    for i in range(num_points):
        for j in range(i+1, num_points):
            D_x[i][j] = ugv_pos[j][0] - ugv_pos[i][0]
            D_x[j][i] = -D_x[i][j]

            D_y[i][j] = ugv_pos[j][1] - ugv_pos[i][1]
            D_y[j][i] = -D_y[i][j]

    return D_x, D_y



# flag= 3 势点分配角度优化


def vehicle_deployment_optimization(vehicle_positions, aim_pos, r_hunt, k_d, k_theta,n_population=50, n_generations=100, mutation_rate=0.01):
    def objective_function(thetas, positions, aim_pos, r_hunt, k_d, k_theta):
        distance_sum = 0
        angle_difference_sum = 0
        for i, theta in enumerate(thetas):
            hunt_pos = np.array([r_hunt * np.cos(theta) + aim_pos[0], r_hunt * np.sin(theta) + aim_pos[1]])
            for pos in positions:
                distance_sum += np.linalg.norm(hunt_pos - pos)  # 计算距离
                angle_difference_sum += np.abs(np.arctan2(hunt_pos[1] - aim_pos[1], hunt_pos[0] - aim_pos[0]) - 
                                            np.arctan2(pos[1] - aim_pos[1], pos[0] - aim_pos[0]))  # 计算角度差
        # 使用 kd 和 ktheta 来控制距离和角度的影响
        return k_d * distance_sum + k_theta * angle_difference_sum

    def create_individual():
        return np.random.uniform(0, 2 * np.pi, len(vehicle_positions))

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(vehicle_positions) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(individual, mutation_rate):
        for i in range(len(vehicle_positions)):
            if random.random() < mutation_rate:
                individual[i] += np.random.normal(0, 0.1)  # small mutation
        return individual

    def selection(population, fitness):
        total_fitness = sum(fitness)
        probability = [f / total_fitness for f in fitness]
        return random.choices(population, weights=probability, k=2)

    # Initialize population
    population = [create_individual() for _ in range(n_population)]
    
    # Evolution process
    for _ in range(n_generations):
        fitness = [objective_function(individual, vehicle_positions, aim_pos, r_hunt, k_d, k_theta) for individual in population]
        next_generation = []
        
        for _ in range(n_population // 2):
            parent1, parent2 = selection(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))
        
        population = np.array(next_generation)
    
    # Find the best individual
    min_fitness_index = np.argmin(fitness)
    best_thetas = population[min_fitness_index]
    
    return best_thetas

# flag= 4 基于合同网的距离、角度分布式优化

def calculate_cost(vehicle, target_angle, aim_pos, r_hunt, k_d, k_theta):
    # 计算到目标角度的旋转成本（以弧度为单位）
    vehicle_pos = np.array([vehicle['x'], vehicle['y']])
    vehicle_angle = vehicle['angle']
    
    angle_diff = abs(vehicle_angle - target_angle)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    rotation_cost = angle_diff
    
    # 计算到目标点的距离成本
    target_pos = aim_pos + r_hunt * np.array([math.cos(target_angle), math.sin(target_angle)])
    distance_cost = np.linalg.norm(vehicle_pos - target_pos)
    
    # 计算总成本
    total_cost = k_d * distance_cost + k_theta * rotation_cost
    return total_cost

def compute_total_cost(vehicles, target_angles, r_hunt, aim_pos, k_d, k_theta):
    total_cost = 0
    for target_angle in target_angles:
        bids = []
        for vehicle in vehicles:
            cost = calculate_cost(vehicle, target_angle, r_hunt, aim_pos, k_d, k_theta)
            bids.append((vehicle, cost))
        
        # 按成本排序，选择成本最低的车辆
        bids.sort(key=lambda x: x[1])
        best_vehicle = bids[0][0]
        
        # 累积总成本
        total_cost += bids[0][1]

    return total_cost

def contract_net_angle_assign(vehicles_pos, aim_pos, r_hunt, k_d, k_theta):
    # 初始化
    n = len(vehicles_pos)
    angle_increment = 2 * np.pi / n
    target_angles = [i * angle_increment for i in range(n)]
    
    vehicles = [{'id': i, 'x': pos[0], 'y': pos[1], 'angle': math.atan2(pos[1] - aim_pos[1], pos[0] - aim_pos[0])} for i, pos in enumerate(vehicles_pos)]

    # 计算每个车辆作为领导者时的总成本，并选择总成本最小的车辆作为领导者
    best_leader = None
    min_total_cost = float('inf')
    for leader in vehicles:
        total_cost = compute_total_cost(vehicles, target_angles, r_hunt, aim_pos, k_d, k_theta)
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_leader = leader

    # 选定领导者，进行任务分配
    assignments = []
    remaining_vehicles = vehicles.copy()

    for target_angle in target_angles:
        bids = []
        for vehicle in remaining_vehicles:
            cost = calculate_cost(vehicle, target_angle, r_hunt, aim_pos, k_d, k_theta)
            bids.append((vehicle, cost))
        
        # 按成本排序，选择成本最低的车辆
        bids.sort(key=lambda x: x[1])
        best_vehicle = bids[0][0]
        
        # 分配环绕点
        assignments.append((best_vehicle['id'], target_angle))
        
        # 从车辆列表中移除已经分配的车辆
        remaining_vehicles = [v for v in remaining_vehicles if v['id'] != best_vehicle['id']]

    # 输出分配的角度
    result = [(assignment[0], assignment[1]) for assignment in assignments]
    result = [assignment[1] for assignment in assignments]

    return result


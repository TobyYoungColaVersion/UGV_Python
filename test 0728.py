# 这一版效果还可以
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 生成随机的UGV位置
np.random.seed(7)  # 设置随机种子以便结果可重复
num_ugvs = 3
vehicle_positions = np.random.rand(num_ugvs, 2) * 10
aim_pos = np.array([5, 5])  # 目标点位置
r_hunt = 1  # 围捕半径
k_d = 0.9  # 距离权重
k_theta = 0.1  # 角度权重

def point_on_circle(center, radius, angle):
    return np.array([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])

def objective_function(thetas, positions, aim_pos, r_hunt, k_d, k_theta):
    distance_sum = 0
    angle_difference_sum = 0
    for i, theta in enumerate(thetas):
        hunt_pos = point_on_circle(aim_pos, r_hunt, theta)
        pos = positions[i]
        distance_sum += np.linalg.norm(hunt_pos - pos)  # 计算距离
        angle_difference_sum += np.abs(np.arctan2(hunt_pos[1] - aim_pos[1], hunt_pos[0] - aim_pos[0]) - 
                                       np.arctan2(pos[1] - aim_pos[1], pos[0] - aim_pos[0]))  # 计算角度差
    return k_d * distance_sum + k_theta * angle_difference_sum

def angle_constraints(thetas):
    n = len(thetas)
    constraints = []
    for i in range(n):
        constraints.append(thetas[i] - thetas[(i-1)%n] - 2*np.pi/n)
    return constraints

def vehicle_deployment_optimization(vehicle_positions, aim_pos, r_hunt, k_d, k_theta):
    num_vehicles = len(vehicle_positions)
    initial_thetas = np.linspace(0, 2 * np.pi, num_vehicles, endpoint=False)
    
    cons = [{'type': 'eq', 'fun': lambda thetas: angle_constraints(thetas)}]
    
    result = minimize(objective_function, initial_thetas, args=(vehicle_positions, aim_pos, r_hunt, k_d, k_theta), bounds=[(0, 2 * np.pi)] * num_vehicles, constraints=cons)
    return result.x

# 进行优化计算
optimal_angles = vehicle_deployment_optimization(vehicle_positions, aim_pos, r_hunt, k_d, k_theta)

# 将弧度制转换为角度制
optimal_angles_degrees = np.degrees(optimal_angles)
print("Optimal angles in degrees:", optimal_angles_degrees)

# 计算分配点
hunt_points = [point_on_circle(aim_pos, r_hunt, angle) for angle in optimal_angles]

# 绘制结果图
plt.figure(figsize=(5, 5))
plt.scatter(vehicle_positions[:, 0], vehicle_positions[:, 1], color='blue', label='UGVs')
plt.scatter(aim_pos[0], aim_pos[1], color='red', label='Aim Point')
plt.scatter([p[0] for p in hunt_points], [p[1] for p in hunt_points], color='green', label='Hunt Points')

# 绘制由UGV指向分配点的箭头
for pos, hunt_pos in zip(vehicle_positions, hunt_points):
    plt.arrow(pos[0], pos[1], hunt_pos[0] - pos[0], hunt_pos[1] - pos[1], head_width=0.3, head_length=0.3, fc='orange', ec='orange')

# 绘制围捕半径
circle = plt.Circle(aim_pos, r_hunt, color='purple', fill=False, linestyle='dashed')
plt.gca().add_artist(circle)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('UGV Deployment Optimization')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.show()

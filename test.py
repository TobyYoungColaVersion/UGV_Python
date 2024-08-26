import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective_function(thetas, positions, aim_pos, r_hunt, k_d, k_theta, theta_escape):
    distance_sum = 0
    print("thetas:", thetas)
    angle_difference_sum = 0
    for i, theta in enumerate(thetas):
        hunt_pos = np.array([r_hunt * np.cos(theta) + aim_pos[0], r_hunt * np.sin(theta) + aim_pos[1]])
        pos = positions[i]
        distance_sum += np.linalg.norm(hunt_pos - pos)
        phi_i = np.arctan2(hunt_pos[1] - pos[1], hunt_pos[0] - pos[0])
        angle_difference_sum += np.abs(phi_i - theta_escape)
    
    return k_d * distance_sum + k_theta * angle_difference_sum

def angle_constraint(thetas):
    n = len(thetas)
    delta = 2 * np.pi / n  # Minimum angular spacing
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(thetas[i] - thetas[j] - delta)
            constraints.append(thetas[j] - thetas[i] - delta)
    return np.array(constraints)

def hunt_points(sim, flag, theta_escape):
    ugv_pos = sim.pur_group.ugv_pos
    sim.hunt_num = len(ugv_pos)
    r_hunt = sim.r_hunt

    if flag == 3:
        initial_angles = np.linspace(0, 2 * np.pi, sim.hunt_num, endpoint=False)
        bounds = [(0, 2 * np.pi)] * sim.hunt_num
        constraints = [{'type': 'ineq', 'fun': angle_constraint}]

        result = minimize(objective_function, initial_angles, args=(ugv_pos, sim.aim_pos, r_hunt, 1, 0, theta_escape), 
                          bounds=bounds, constraints=constraints)

        optimal_angles = result.x
        sim.hunt_ps = np.array([point_on_circle((sim.aim_pos[0], sim.aim_pos[1]), sim.r_hunt, angle) for angle in optimal_angles]).tolist()

        for i in range(sim.hunt_num):
            if i not in sim.hunt_ps_store:
                sim.hunt_ps_store[i] = []
            sim.hunt_ps_store[i].append(sim.hunt_ps[i])

def point_on_circle(center, radius, angle):
    return [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)]

def plot_positions(ugv_pos, hunt_ps, aim_pos, theta_escape, r_hunt):
    plt.figure(figsize=(8, 8))
    plt.scatter(*zip(*ugv_pos), color='blue', label='Pursuers')
    plt.scatter(*zip(*hunt_ps), color='red', label='Target Points')
    plt.scatter(aim_pos[0], aim_pos[1], color='green', label='Aim Position')
    for pos, hunt in zip(ugv_pos, hunt_ps):
        plt.plot([pos[0], hunt[0]], [pos[1], hunt[1]], 'gray', linestyle='--')
    plt.quiver(aim_pos[0], aim_pos[1], np.cos(theta_escape), np.sin(theta_escape), color='orange', scale=10, label='Escape Direction')
    circle = plt.Circle(aim_pos, r_hunt, color='purple', fill=False, linestyle='--', label='Hunt Radius')
    plt.gca().add_artist(circle)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pursuers and Target Points Visualization')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example data
ugv_pos = np.array([[0, -10], [-5, 8], [-6, 7]])
aim_pos = np.array([4, 4])
r_hunt = 3
theta_escape = np.pi / 4  # Escape direction is 45 degrees

# Initialize hunt_points object
class HuntSimulation:
    def __init__(self, pur_group, r_hunt, aim_pos):
        self.pur_group = pur_group
        self.r_hunt = r_hunt
        self.aim_pos = aim_pos
        self.hunt_ps_store = {}
        self.hunt_ps = []

class PurGroup:
    def __init__(self, ugv_pos):
        self.ugv_pos = ugv_pos

pur_group = PurGroup(ugv_pos)
sim = HuntSimulation(pur_group, r_hunt, aim_pos)
hunt_points(sim, flag=3, theta_escape=theta_escape)

# Plot the positions
plot_positions(ugv_pos, sim.hunt_ps, aim_pos, theta_escape, r_hunt)


  
    
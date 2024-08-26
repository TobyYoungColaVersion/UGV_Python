import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, linear_sum_assignment
import os


# Parameters
n = 3  # Number of pursuers
r_hunt = 10  # Capture radius
k1, k2, k3 = 0.7, 0.1, 0.2  # Weights for the objective function
x_target, y_target = 0, 0  # Target position

# Randomly generate initial positions and directions for pursuers and direction for the evader
np.random.seed(None)
x_pursuers = np.random.uniform(-20, 20, n)
y_pursuers = np.random.uniform(-20, 20, n)
phi_pursuers = np.random.uniform(0, 2 * np.pi, n)
phi_evader = np.random.uniform(0, 2 * np.pi)  # Evader direction angle

# Objective function
def objective(angles):
    target_points = np.array([(x_target + r_hunt * np.cos(angle), 
                               y_target + r_hunt * np.sin(angle)) for angle in angles])
    
    distance_matrix = np.array([[np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2) 
                                 for (x_t, y_t) in target_points] for (x_p, y_p) in zip(x_pursuers, y_pursuers)])
    
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    assigned_points = target_points[col_ind]
    
    total_distance = distance_matrix[row_ind, col_ind].sum()
    phi2 = np.array([np.arctan2(y_t - y_p, x_t - x_p) 
                     for (x_p, y_p), (x_t, y_t) in zip(zip(x_pursuers, y_pursuers), assigned_points)])
    total_angle_change = np.sum(np.abs(phi2 - phi_pursuers))
    total_delta_angle = np.sum(np.abs(phi2 - phi_evader))
    
    F = k1 * total_distance + k2 * total_angle_change + k3 * total_delta_angle
    return F

# Constraint function for angle differences
def angle_difference_constraint(angles):
    diff = np.diff(np.sort(angles))
    diff = np.append(diff, 2 * np.pi - np.sum(diff))
    return diff - (2 * np.pi / n)

# Initial angles
initial_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

# Nonlinear constraints
nonlinear_constraint = NonlinearConstraint(angle_difference_constraint, 0, 0)

# Optimization
result = minimize(objective, initial_angles, bounds=[(0, 2 * np.pi)] * n, constraints=[nonlinear_constraint])

# Optimal solution
optimal_angles = result.x % (2 * np.pi)

# Calculate assigned target points
optimal_target_points = np.array([(x_target + r_hunt * np.cos(angle), 
                                   y_target + r_hunt * np.sin(angle)) for angle in optimal_angles])

# Calculate distance matrix
distance_matrix = np.array([[np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2) 
                             for (x_t, y_t) in optimal_target_points] for (x_p, y_p) in zip(x_pursuers, y_pursuers)])

# Use Hungarian algorithm for optimal assignment
row_ind, col_ind = linear_sum_assignment(distance_matrix)
assigned_points = optimal_target_points[col_ind]

# Calculate total distance, angle change, and angle difference
total_distance = distance_matrix[row_ind, col_ind].sum()
phi2 = np.array([np.arctan2(y_t - y_p, x_t - x_p) 
                 for (x_p, y_p), (x_t, y_t) in zip(zip(x_pursuers, y_pursuers), assigned_points)])
total_angle_change = np.sum(np.abs(phi2 - phi_pursuers))
total_delta_angle = np.sum(np.abs(phi2 - phi_evader))

# Convert angles from radians to degrees
phi_pursuers_deg = np.degrees(phi_pursuers) % 360
phi2_deg = np.degrees(phi2) % 360
phi_evader_deg = np.degrees(phi_evader) % 360

# Calculate the objective function
F = k1 * total_distance + k2 * total_angle_change + k3 * total_delta_angle

# Output the optimization results in degrees
print(f"Optimization Objective F = {F:.2f}")
print(f"Total Distance = {total_distance:.2f}")
print(f"Total Angle Change = {total_angle_change:.2f}")
print(f"Total Angle Difference = {total_delta_angle:.2f}")
print("\nInitial Pursuer Directions (Degrees):", phi_pursuers_deg)
print("Assigned Target Point Angles (Degrees):", phi2_deg)
print(f"Evader Direction (Degrees): {phi_evader_deg:.2f}")
print("Optimal Target Point Angles (Degrees):", np.degrees(optimal_angles) % 360)

# Plotting
plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Draw target and capture radius
circle = plt.Circle((x_target, y_target), r_hunt, color='r', fill=False, linestyle='--', label='Capture Radius')
ax.add_patch(circle)
plt.scatter([x_target], [y_target], color='r', label='Target')

# Draw pursuers, arrows pointing to assigned target points, and their directions
for i, ((x_p, y_p), (x_t, y_t), phi_p) in enumerate(zip(zip(x_pursuers, y_pursuers), assigned_points, phi_pursuers)):
    # Calculate direction vector pointing to assigned target
    direction = np.array([x_t - x_p, y_t - y_p])
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm  # Normalize
    # Ensure arrow points exactly at the target point
    plt.scatter(x_p, y_p, color='b', label='Pursuer' if i == 0 else "")
    plt.arrow(x_p, y_p, direction[0] * 0.85 * norm, direction[1] * 0.85 * norm, head_width=0.4, head_length=0.5, fc='b', ec='b')
    # Draw pursuer's direction
    orientation_vector = np.array([np.cos(phi_p), np.sin(phi_p)])
    plt.arrow(x_p, y_p, orientation_vector[0] * 1.5, orientation_vector[1] * 1.5, head_width=0.5, head_length=0.8, fc='purple', ec='purple', linestyle='dotted')
    # Label pursuer's direction
    plt.text(x_p + orientation_vector[0] * 3, y_p + orientation_vector[1] * 3, f'P{i+1}', color='purple', fontsize=5, weight='bold')

# Draw target points
for i, (x_t, y_t) in enumerate(optimal_target_points):
    plt.scatter(x_t, y_t, color='g', marker='x', label='Target Point' if i == 0 else "")

# Draw angle division lines
for angle in optimal_angles:
    x_a = x_target + r_hunt * np.cos(angle)
    y_a = x_target + r_hunt * np.sin(angle)
    plt.plot([x_target, x_a], [y_target, y_a], linestyle='--', color='gray')

# Draw evader's direction
evader_vector = np.array([np.cos(phi_evader), np.sin(phi_evader)])
plt.arrow(x_target, y_target, evader_vector[0] * 0.2 * r_hunt, evader_vector[1] * 0.2 * r_hunt, head_width=0.5, head_length=0.7, fc='orange', ec='orange', linestyle='dotted', label='Evader Direction')

plt.legend(fontsize=5)
plt.xlabel('X', fontsize=5)
plt.ylabel('Y', fontsize=5)
plt.title('Potential Point Assignment', fontsize=12)
plt.grid(True)


# Display the plot
plt.show()

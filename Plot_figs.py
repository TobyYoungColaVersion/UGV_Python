from typing import List
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from Other_Func import apollonius_circle
from Sim_Class import UGV, UGV_Group, Hunting_Tar, Obstacles  
from Dynamic_Assignment import Dynamic_Assignment
import os

def plot_dynamic_traces_multi_apollo(groups: List[UGV_Group], targets: List[Hunting_Tar], obstacle: Obstacles, r_hunt, dt, save_path: str, m: int):

    # Initialize traces and IDs
    UGV_Trace = []
    UGV_start_times = []
    UGV_ids = []
    UGV_aim_Tars = []
    for group in groups:
        for ugv in group.ugvs:
            UGV_Trace.append(ugv.path)
            UGV_start_times.append(ugv.start_time)
            UGV_ids.append(ugv.id)
            UGV_aim_Tars.append(ugv.ugv_aim_Tar)

    # Sort UGVs by their IDs
    sorted_indices = np.argsort(UGV_ids)
    UGV_Trace = [UGV_Trace[i] for i in sorted_indices]
    UGV_start_times = [UGV_start_times[i] for i in sorted_indices]
    UGV_ids = [UGV_ids[i] for i in sorted_indices]
    UGV_aim_Tars = [UGV_aim_Tars[i] for i in sorted_indices]

    Target_trace = [target.trace for target in targets]
    Target_start_times = [target.start_time for target in targets]
    Target_ids = [target.Tar_id for target in targets]

    all_traces = UGV_Trace + Target_trace
    all_points = np.vstack(all_traces)

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    max_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Plot setup
    num_points = max([len(trace) for trace in UGV_Trace + Target_trace])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)

    # Initialize line objects with UGV IDs in legend
    ugv_lines = [ax.plot([], [], lw=2, linestyle='--', label=f'UGV_{ugv_id}')[0] for ugv_id in UGV_ids]
    target_lines = [ax.plot([], [], lw=2, label=f'Target_{i}')[0] for i in range(len(Target_trace))]
    ax.legend()

    # Initialize circle objects
    circle_hunts = [plt.Circle((0, 0), 0, color='black', fill=False, linestyle='--') for _ in targets]
    for circle_hunt in circle_hunts:
        ax.add_patch(circle_hunt)

    # Initialize scatter objects
    ugv_scatters = [ax.scatter([], [], s=50) for _ in range(len(UGV_Trace))]
    target_scatters = [ax.scatter([], [], s=50, color='red') for _ in range(len(Target_trace))]

    # Initialize time text object
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Initialize Apollonius circle objects
    apollonius_circles = [plt.Circle((0, 0), 0, color='blue', fill=False, linestyle='-.', label=f'Apollo_{i}') for i in range(len(UGV_Trace))]
    for circle_0 in apollonius_circles:
        ax.add_patch(circle_0)
        ax.legend()

    # Initialize obstacle scatter object
    obstacle_scatter = ax.scatter([], [], s=100, color='orange', label='Obstacles')
    ax.legend()

    def init():
        for line in ugv_lines + target_lines:
            line.set_data([], [])
        for scatter in ugv_scatters + target_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for circle in apollonius_circles:
            circle.center = (0, 0)
            circle.set_radius(0)
        for circle_hunt in circle_hunts:
            circle_hunt.center = (0, 0)
        time_text.set_text('')
        obstacle_scatter.set_offsets(np.empty((0, 2)))
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text] + [obstacle_scatter]

    def update(frame):
        for line, trace, start_time in zip(ugv_lines, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(ugv_scatters, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for line, trace, start_time in zip(target_lines, Target_trace, Target_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(target_scatters, Target_trace, Target_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for target, circle_hunt, start_time in zip(Target_trace, circle_hunts, Target_start_times):
            if frame >= start_time:
                circle_hunt.center = (target[frame - start_time, 0], target[frame - start_time, 1])
                circle_hunt.set_radius(r_hunt)
        
        time_text.set_text(f'Time: {frame * dt:.2f}s')
        
        # Update Apollonius circles
        for i, ap_circle in enumerate(apollonius_circles):
            if frame >= UGV_start_times[i]:
                current_target_id = UGV_aim_Tars[i][frame - UGV_start_times[i]]
                if current_target_id in Target_ids:
                    target_index = Target_ids.index(current_target_id)
                    if frame >= Target_start_times[target_index]:
                        center_x, center_y, radius = apollonius_circle(
                            UGV_Trace[i][frame - UGV_start_times[i]], 1.78, 
                            Target_trace[target_index][frame - Target_start_times[target_index]], 2
                        )
                        ap_circle.center = (center_x, center_y)
                        ap_circle.set_radius(radius)

        # Update obstacle positions
        obstacle_scatter.set_offsets(np.array(obstacle.positions))
         # Save frame every m frames
        if frame % m == 0:
            png_filename = os.path.join(save_path, f'frame_{frame}.png')
            eps_filename = os.path.join(save_path, f'frame_{frame}.eps')

            fig.savefig(png_filename, format='png')
            fig.savefig(eps_filename, format='eps')
    
        return ugv_lines + target_lines + ugv_scatters + target_scatters + circle_hunts + [time_text]

    # Create directory if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text] + [obstacle_scatter]
    
    interval = 50

    ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=interval)
        # Save the animation as a GIF
    # ani.save('dynamic_traces.gif', writer='pillow', fps=20)
    plt.show()
   

def plot_dynamic_traces_multi_ParamTest(groups: List[UGV_Group], targets: List[Hunting_Tar], obstacle: Obstacles, r_hunt, dt, save_path: str, m: int):

    # Initialize traces and IDs
    UGV_Trace = []
    UGV_start_times = []
    UGV_ids = []
    UGV_aim_Tars = []
    for group in groups:
        for ugv in group.ugvs:
            UGV_Trace.append(ugv.path)
            UGV_start_times.append(ugv.start_time)
            UGV_ids.append(ugv.id)
            UGV_aim_Tars.append(ugv.ugv_aim_Tar)

    # Sort UGVs by their IDs
    sorted_indices = np.argsort(UGV_ids)
    UGV_Trace = [UGV_Trace[i] for i in sorted_indices]
    UGV_start_times = [UGV_start_times[i] for i in sorted_indices]
    UGV_ids = [UGV_ids[i] for i in sorted_indices]
    UGV_aim_Tars = [UGV_aim_Tars[i] for i in sorted_indices]

    Target_trace = [target.trace for target in targets]
    Target_start_times = [target.start_time for target in targets]
    Target_ids = [target.Tar_id for target in targets]

    all_traces = UGV_Trace + Target_trace
    all_points = np.vstack(all_traces)

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    max_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Plot setup
    num_points = max([len(trace) for trace in UGV_Trace + Target_trace])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min - 8, x_max + 8)
    ax.set_ylim(x_min - 8, x_max + 8)

    # Initialize line objects with UGV IDs in legend
    ugv_lines = [ax.plot([], [], lw=2, linestyle='--', label=f'P{ugv_id}')[0] for ugv_id in UGV_ids]
    target_lines = [ax.plot([], [], lw=2, label=f'E{i}')[0] for i in range(len(Target_trace))]
    ax.legend()

    # Initialize circle objects
    circle_hunts = [plt.Circle((0, 0), 0, color='black', fill=False, linestyle='--') for _ in targets]
    for circle_hunt in circle_hunts:
        ax.add_patch(circle_hunt)

    # Initialize scatter objects
    ugv_scatters = [ax.scatter([], [], s=50) for _ in range(len(UGV_Trace))]
    target_scatters = [ax.scatter([], [], s=50, color='red') for _ in range(len(Target_trace))]

    # Initialize time text object
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Initialize Apollonius circle objects
    apollonius_circles = [plt.Circle((0, 0), 0, color='blue', fill=False, linestyle='-.') for _ in range(len(UGV_Trace))]
    for circle_0 in apollonius_circles:
        ax.add_patch(circle_0)
        ax.legend()

    ax.legend()

    def init():
        for line in ugv_lines + target_lines:
            line.set_data([], [])
        for scatter in ugv_scatters + target_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for circle in apollonius_circles:
            circle.center = (0, 0)
            circle.set_radius(0)
        for circle_hunt in circle_hunts:
            circle_hunt.center = (0, 0)
        time_text.set_text('')
       
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text]

    def update(frame):
        for line, trace, start_time in zip(ugv_lines, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(ugv_scatters, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for line, trace, start_time in zip(target_lines, Target_trace, Target_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(target_scatters, Target_trace, Target_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for target, circle_hunt, start_time in zip(Target_trace, circle_hunts, Target_start_times):
            if frame >= start_time:
                circle_hunt.center = (target[frame - start_time, 0], target[frame - start_time, 1])
                circle_hunt.set_radius(r_hunt)
        
        time_text.set_text(f'Time: {frame * dt:.2f}s')
        
        # Update Apollonius circles
        for i, ap_circle in enumerate(apollonius_circles):
            if frame >= UGV_start_times[i]:
                current_target_id = UGV_aim_Tars[i][frame - UGV_start_times[i]]
                if current_target_id in Target_ids:
                    target_index = Target_ids.index(current_target_id)
                    if frame >= Target_start_times[target_index]:
                        center_x, center_y, radius = apollonius_circle(
                            UGV_Trace[i][frame - UGV_start_times[i]], 1.81, 
                            Target_trace[target_index][frame - Target_start_times[target_index]], 2
                        )
                        ap_circle.center = (center_x, center_y)
                        ap_circle.set_radius(radius)

         # Save frame every m frames
        if frame % m == 0:
            png_filename = os.path.join(save_path, f'frame_{frame}_apollo.png')
            eps_filename = os.path.join(save_path, f'frame_{frame}_apollo.eps')

            fig.savefig(png_filename, format='png')
            fig.savefig(eps_filename, format='eps')
    
        return ugv_lines + target_lines + ugv_scatters + target_scatters + circle_hunts + [time_text]

    # Create directory if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # Update obstacle positions
        
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text]
    
    interval = 50

    ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=interval)
        # Save the animation as a GIF
    # ani.save('dynamic_traces.gif', writer='pillow', fps=20)
    plt.show()
   

def plot_dynamic_traces_multi_discoupled_gif(groups: List[UGV_Group], targets: List[Hunting_Tar], obstacle: Obstacles, r_hunt, dt):

    # Initialize traces and IDs
    UGV_Trace = []
    UGV_start_times = []
    UGV_ids = []
    UGV_aim_Tars = []
    for group in groups:
        for ugv in group.ugvs:
            UGV_Trace.append(ugv.path)
            UGV_start_times.append(ugv.start_time)
            UGV_ids.append(ugv.id)
            UGV_aim_Tars.append(ugv.ugv_aim_Tar)

    # Sort UGVs by their IDs
    sorted_indices = np.argsort(UGV_ids)
    UGV_Trace = [UGV_Trace[i] for i in sorted_indices]
    UGV_start_times = [UGV_start_times[i] for i in sorted_indices]
    UGV_ids = [UGV_ids[i] for i in sorted_indices]
    UGV_aim_Tars = [UGV_aim_Tars[i] for i in sorted_indices]

    Target_trace = [target.trace for target in targets]
    Target_start_times = [target.start_time for target in targets]
    Target_ids = [target.Tar_id for target in targets]

    all_traces = UGV_Trace + Target_trace
    all_points = np.vstack(all_traces)

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    max_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Plot setup
    num_points = max([len(trace) for trace in UGV_Trace + Target_trace])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min - 1000, x_max + 1000)
    ax.set_ylim(y_min - 1000, y_max + 1000)

    # Initialize line objects with UGV IDs in legend
    ugv_lines = [ax.plot([], [], lw=2, linestyle='--', label=f'UGV_{ugv_id}')[0] for ugv_id in UGV_ids]
    target_lines = [ax.plot([], [], lw=2, label=f'Target_{i}')[0] for i in range(len(Target_trace))]
    ax.legend()

    # Initialize circle objects
    circle_hunts = [plt.Circle((0, 0), 0, color='black', fill=False, linestyle='--') for _ in targets]
    for circle_hunt in circle_hunts:
        ax.add_patch(circle_hunt)

    # Initialize scatter objects
    ugv_scatters = [ax.scatter([], [], s=50) for _ in range(len(UGV_Trace))]
    target_scatters = [ax.scatter([], [], s=50, color='red') for _ in range(len(Target_trace))]

    # Initialize time text object
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Initialize Apollonius circle objects
    apollonius_circles = [plt.Circle((0, 0), 0, color='blue', fill=False, linestyle='-.', label=f'Apollo_{i}') for i in range(len(UGV_Trace))]
    for circle_0 in apollonius_circles:
        ax.add_patch(circle_0)
        ax.legend()

    # Initialize obstacle scatter object
    obstacle_scatter = ax.scatter([], [], s=100, color='orange', label='Obstacles')
    ax.legend()

    def init():
        for line in ugv_lines + target_lines:
            line.set_data([], [])
        for scatter in ugv_scatters + target_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for circle in apollonius_circles:
            circle.center = (0, 0)
            circle.set_radius(0)
        for circle_hunt in circle_hunts:
            circle_hunt.center = (0, 0)
        time_text.set_text('')
        obstacle_scatter.set_offsets(np.empty((0, 2)))
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text] + [obstacle_scatter]

    def update(frame):
        for line, trace, start_time in zip(ugv_lines, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(ugv_scatters, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for line, trace, start_time in zip(target_lines, Target_trace, Target_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(target_scatters, Target_trace, Target_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for target, circle_hunt, start_time in zip(Target_trace, circle_hunts, Target_start_times):
            if frame >= start_time:
                circle_hunt.center = (target[frame - start_time, 0], target[frame - start_time, 1])
                circle_hunt.set_radius(r_hunt)
        
        time_text.set_text(f'Time: {frame * dt:.2f}s')
        
        # Update Apollonius circles
        for i, ap_circle in enumerate(apollonius_circles):
            if frame >= UGV_start_times[i]:
                current_target_id = UGV_aim_Tars[i][frame - UGV_start_times[i]]
                if current_target_id in Target_ids:
                    target_index = Target_ids.index(current_target_id)
                    if frame >= Target_start_times[target_index]:
                        center_x, center_y, radius = apollonius_circle(
                            UGV_Trace[i][frame - UGV_start_times[i]], 1.73, 
                            Target_trace[target_index][frame - Target_start_times[target_index]], 2
                        )
                        ap_circle.center = (center_x, center_y)
                        ap_circle.set_radius(radius)

        # Update obstacle positions
        obstacle_scatter.set_offsets(np.array(obstacle.positions))
        
        return ugv_lines + target_lines + ugv_scatters + target_scatters + apollonius_circles + circle_hunts + [time_text] + [obstacle_scatter]
    
    interval = 50

    ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=interval)

    plt.show()



def plot_dynamic_traces_multi_ParamTest_no_apollo(groups: List[UGV_Group], targets: List[Hunting_Tar], obstacle: Obstacles, r_hunt, dt, save_path: str, m: int):

    # Initialize traces and IDs
    UGV_Trace = []
    UGV_start_times = []
    UGV_ids = []
    UGV_aim_Tars = []
    for group in groups:
        for ugv in group.ugvs:
            UGV_Trace.append(ugv.path)
            UGV_start_times.append(ugv.start_time)
            UGV_ids.append(ugv.id)
            UGV_aim_Tars.append(ugv.ugv_aim_Tar)

    # Sort UGVs by their IDs
    sorted_indices = np.argsort(UGV_ids)
    UGV_Trace = [UGV_Trace[i] for i in sorted_indices]
    UGV_start_times = [UGV_start_times[i] for i in sorted_indices]
    UGV_ids = [UGV_ids[i] for i in sorted_indices]
    UGV_aim_Tars = [UGV_aim_Tars[i] for i in sorted_indices]

    Target_trace = [target.trace for target in targets]
    Target_start_times = [target.start_time for target in targets]
    Target_ids = [target.Tar_id for target in targets]

    all_traces = UGV_Trace + Target_trace
    all_points = np.vstack(all_traces)

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    max_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Plot setup
    num_points = max([len(trace) for trace in UGV_Trace + Target_trace])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)

    # Initialize line objects with UGV IDs in legend
    ugv_lines = [ax.plot([], [], lw=2, linestyle='--', label=f'P{ugv_id+1}')[0] for ugv_id in UGV_ids]
    target_lines = [ax.plot([], [], lw=2, label=f'E{i}')[0] for i in range(len(Target_trace))]
    ax.legend()

    # Initialize circle objects
    circle_hunts = [plt.Circle((0, 0), 0, color='black', fill=False, linestyle='--') for _ in targets]
    for circle_hunt in circle_hunts:
        ax.add_patch(circle_hunt)

    # Initialize scatter objects
    ugv_scatters = [ax.scatter([], [], s=50) for _ in range(len(UGV_Trace))]
    target_scatters = [ax.scatter([], [], s=50, color='red') for _ in range(len(Target_trace))]

    # Initialize time text object
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    ax.legend()

    def init():
        for line in ugv_lines + target_lines:
            line.set_data([], [])
        for scatter in ugv_scatters + target_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for circle_hunt in circle_hunts:
            circle_hunt.center = (0, 0)
        time_text.set_text('')
        return ugv_lines + target_lines + ugv_scatters + target_scatters+ circle_hunts + [time_text]

    def update(frame):
        for line, trace, start_time in zip(ugv_lines, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(ugv_scatters, UGV_Trace, UGV_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for line, trace, start_time in zip(target_lines, Target_trace, Target_start_times):
            if frame >= start_time:
                line.set_data(trace[:frame - start_time, 0], trace[:frame - start_time, 1])
        for scatter, trace, start_time in zip(target_scatters, Target_trace, Target_start_times):
            if frame >= start_time:
                scatter.set_offsets(trace[frame - start_time, :2])
        
        for target, circle_hunt, start_time in zip(Target_trace, circle_hunts, Target_start_times):
            if frame >= start_time:
                circle_hunt.center = (target[frame - start_time, 0], target[frame - start_time, 1])
                circle_hunt.set_radius(r_hunt)
        
        time_text.set_text(f'Time: {frame * dt:.2f}s')

        # Save frame every m frames
        if frame % m == 0:
            png_filename = os.path.join(save_path, f'frame_{frame}.png')
            eps_filename = os.path.join(save_path, f'frame_{frame}.eps')

            fig.savefig(png_filename, format='png')
            fig.savefig(eps_filename, format='eps')
    
        return ugv_lines + target_lines + ugv_scatters + target_scatters + circle_hunts + [time_text]

    # Create directory if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    interval = 50
    ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=interval)
    plt.show()
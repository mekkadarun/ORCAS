import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from motion_models.real_quadrotor import RealQuadrotor
from motion_models.real_obstacle import GMMObstacle
from control.real_mpc import RealCVaRGMMMPC_3D

def draw_confidence_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw a covariance ellipse for visualization."""
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(*mean)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_risk_heatmap(ax, obstacles, cvar_controller, x_range, y_range, resolution=0.2):
    """Generate and draw a risk heatmap for visualization."""
    risk_map, X, Y = cvar_controller.cvar_avoidance.generate_risk_map(
        x_range, y_range, resolution, obstacles
    )
    
    # Create a custom colormap with transparency
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)**0.5  # Alpha channel adjustment
    my_cmap = colors.ListedColormap(my_cmap)
    
    heatmap = ax.pcolormesh(X, Y, risk_map, cmap=my_cmap, 
                          vmin=0, vmax=1, shading='auto')
    
    return heatmap

def run_simulation():
    """Run the simulation and create an animation with the 3D quadrotor model."""
    # Initialize components
    # Initial state: [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
    initial_state = np.zeros(12)
    initial_state[:3] = np.array([0.0, 0.0, 0.0])  # Initial position
    
    quadrotor = RealQuadrotor(initial_state)
    
    # Create obstacles (now defined in 3D)
    # Goal position (x, y, z)
    goal = np.array([7.0, 7.0, 3.0])  # 3D goal with a clear height component
    
    # Create obstacles specifically at the goal's z-height to create a challenge
    obstacles = [
        GMMObstacle((2.0, 2.0, goal[2]), radius=0.3, n_components=2, is_3d=True),
        GMMObstacle((4.0, 4.0, goal[2]), radius=0.4, n_components=2, is_3d=True),
        GMMObstacle((2.70, 1.0, goal[2]), radius=0.3, n_components=2, is_3d=True),
        GMMObstacle((5.0, 5.0, goal[2]), radius=0.35, n_components=2, is_3d=True),
        GMMObstacle((6.0, 6.0, goal[2]), radius=0.25, n_components=2, is_3d=True)
    ]

    # Initialize CVaR-based MPC controller
    mpc = RealCVaRGMMMPC_3D(horizon=10, dt=0.1, quad_radius=0.3, confidence_level=0.97)

    # Tracking variables
    step_count = 0
    total_cost = 0.0
    actual_trajectory = []
    obstacle_trajectories = [[] for _ in obstacles]

    # Pre-train GMM models with 3D movement patterns that maintain altitude
    movement_patterns = [
        ([0.1, 0.2, 0.0], [[0.02, 0.005, 0.001], [0.005, 0.02, 0.001], [0.001, 0.001, 0.005]]),
        ([0.1, -0.1, 0.0], [[0.03, -0.01, 0.0005], [-0.01, 0.03, 0.0005], [0.0005, 0.0005, 0.003]]),
        ([-0.1, 0.15, 0.0], [[0.04, 0.02, 0.0008], [0.02, 0.04, 0.0008], [0.0008, 0.0008, 0.004]])
    ]

    # Generate initial movement data
    for i, obs in enumerate(obstacles):
        pattern_idx = i % len(movement_patterns)
        mean, cov = movement_patterns[pattern_idx]
        
        for _ in range(30):
            movement = np.random.multivariate_normal(mean=mean, cov=cov)
            obs.movement_history.append(movement)
            obs.ambiguity_set.add_movement_data(movement)
        
        for _ in range(5):
            outlier = np.random.multivariate_normal(
                mean=[-m for m in mean],
                cov=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
            )
            obs.movement_history.append(outlier)
            obs.ambiguity_set.add_movement_data(outlier)
        
        # Initialize GMM and ambiguity set
        obs._fit_gmm()
        obs.ambiguity_set.update_mixture_model()
        
        # Record initial position
        obstacle_trajectories[i].append(obs.get_position().copy())

    # Create figure for visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Create 3D subplot for 3D visualization
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Ensure consistent scale across axes
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio in 3D 
    
    # Set consistent limits 
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-1, 10)
    ax1.set_zlim(-1, 5)
    ax1.set_title("3D View - Quadrotor Motion Control", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_zlabel("Z Position (m)")
    
    # Add gridlines for better depth perception
    ax1.grid(True)
    
    # Set better view angle
    ax1.view_init(elev=30, azim=45)
    
    # Create 2D subplot for top-down view with risk map
    ax2 = fig.add_subplot(122, projection='3d')  # Now this is also 3D
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-1, 10)
    ax2.set_zlim(-1, 5)
    ax2.view_init(elev=90, azim=0)  # Top-down view
    ax2.set_title("Top-Down View with Height Information", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_zlabel("Z Position (m)")

    # Plot static elements
    # 3D goal in both plots
    ax1.scatter(*goal, s=200, c='green', marker='*', label='Goal', zorder=10)
    ax2.scatter(*goal, s=200, c='green', marker='*', label='Goal', zorder=10)

    # Dynamic elements for 3D plot
    quad_marker_3d = ax1.plot([], [], [], 'o', color='blue', markersize=10, label='Quadrotor')[0]
    traj_line_3d = ax1.plot([], [], [], '--', color='blue', label='Trajectory')[0]
    
    # Draw quadrotor orientation lines
    roll_line = ax1.plot([], [], [], '-', color='red', linewidth=2, label='Roll Axis')[0]
    pitch_line = ax1.plot([], [], [], '-', color='green', linewidth=2, label='Pitch Axis')[0]
    yaw_line = ax1.plot([], [], [], '-', color='blue', linewidth=2, label='Yaw Axis')[0]
    
    # Line from quadrotor to goal to show true distance
    goal_dist_line = ax1.plot([], [], [], 'g:', linewidth=1, alpha=0.5, label='Distance to Goal')[0]
    
    # Dynamic elements for 2D plot
    quad_marker_2d = ax2.plot([], [], [], 'o', color='blue', markersize=10, label='Quadrotor', zorder=10)[0]
    traj_line_2d = ax2.plot([], [], [], '--', color='blue', label='Trajectory', zorder=5)[0]

    # Obstacle trajectories 
    obs_traj_lines = []
    for i in range(len(obstacles)):
        line, = ax2.plot([], [], [], alpha=0.6, zorder=4)
        obs_traj_lines.append(line)

    # Status text display
    status_text = ax2.text2D(
        0.02, 0.98, '', 
        transform=ax2.transAxes, 
        verticalalignment='top',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        zorder=20
    )

    # Obstacle markers
    obs_markers_3d = []  # 3D markers in first view
    obs_markers_2d = []  # 3D markers in top view

    # Initialize visualization
    for i, obs in enumerate(obstacles):
        # 3D visualization - main view
        sphere = ax1.plot([],[],[], 'o', color=f'C{i+1}', markersize=20, alpha=0.7)[0]
        obs_markers_3d.append(sphere)
        
        # 3D visualization - top view
        sphere2 = ax2.plot([],[],[], 'o', color=f'C{i+1}', markersize=20, alpha=0.7)[0]
        obs_markers_2d.append(sphere2)
    
    # Add helper lines for height visualization in top-down view
    height_lines = []
    for i in range(len(obstacles)):
        line, = ax2.plot([], [], [], '-', color=f'C{i+1}', linewidth=1, alpha=0.5)
        height_lines.append(line)
    
    # Add quadrotor height line in top-down view
    quad_height_line = ax2.plot([], [], [], '-', color='blue', linewidth=1, alpha=0.5)[0]

    def draw_orientation(position, orientation, scale=0.5):
        """Draw orientation axes of the quadrotor"""
        roll, pitch, yaw = orientation
        
        # Create rotation matrix from Euler angles
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Roll axis (rotated x-axis)
        roll_dir = np.array([
            cos_yaw * cos_pitch,
            sin_yaw * cos_pitch,
            -sin_pitch
        ]) * scale
        
        # Pitch axis (rotated y-axis)
        pitch_dir = np.array([
            -cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
            -sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
            -cos_pitch * sin_roll
        ]) * scale
        
        # Yaw axis (rotated z-axis)
        yaw_dir = np.array([
            -cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            -sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            -cos_pitch * cos_roll
        ]) * scale
        
        return roll_dir, pitch_dir, yaw_dir

    def init():
        """Initialize animation elements."""
        quad_marker_3d.set_data([], [])
        quad_marker_3d.set_3d_properties([])
        traj_line_3d.set_data([], [])
        traj_line_3d.set_3d_properties([])
        
        roll_line.set_data([], [])
        roll_line.set_3d_properties([])
        pitch_line.set_data([], [])
        pitch_line.set_3d_properties([])
        yaw_line.set_data([], [])
        yaw_line.set_3d_properties([])
        
        goal_dist_line.set_data([], [])
        goal_dist_line.set_3d_properties([])
        
        quad_marker_2d.set_data([], [])
        quad_marker_2d.set_3d_properties([])
        traj_line_2d.set_data([], [])
        traj_line_2d.set_3d_properties([])
        
        quad_height_line.set_data([], [])
        quad_height_line.set_3d_properties([])
        
        for line in obs_traj_lines:
            line.set_data([], [])
            line.set_3d_properties([])
            
        for line in height_lines:
            line.set_data([], [])
            line.set_3d_properties([])
            
        status_text.set_text('Starting simulation...')
        
        return ([quad_marker_3d, traj_line_3d, roll_line, pitch_line, yaw_line, 
                goal_dist_line, quad_marker_2d, traj_line_2d, quad_height_line, 
                status_text] + obs_markers_3d + obs_markers_2d + 
                obs_traj_lines + height_lines)

    def update(frame):
        """Update animation for each frame."""
        nonlocal step_count, total_cost
        step_count += 1
        
        current_state = quadrotor.get_state()
        actual_trajectory.append(current_state[:3].copy())
        
        position = current_state[:3]  # 3D position
        
        # Calculate ACTUAL distance to goal in 3D space
        distance_to_goal = np.linalg.norm(position - goal)
        
        # Split into horizontal and vertical distances for better understanding
        horiz_distance = np.linalg.norm(position[:2] - goal[:2])
        vert_distance = abs(position[2] - goal[2])
        
        # Cap total cost to prevent huge numbers
        total_cost = min(total_cost, 10000)
        
        status_text.set_text(f"Step: {step_count} | 3D Distance: {distance_to_goal:.2f}m | "
                            f"Horiz: {horiz_distance:.2f}m | Vert: {vert_distance:.2f}m | "
                            f"Cost: {total_cost:.2f}")
        
        # Check if goal is reached - use true 3D distance
        if distance_to_goal < 0.5:
            print(f"Goal reached at step {step_count}!")
            print(f"Total cost: {total_cost:.2f}")
            print(f"Final position: {position}")
            print(f"Goal position: {goal}")
            print(f"Horizontal error: {horiz_distance:.2f}m")
            print(f"Vertical error: {vert_distance:.2f}m")
            ani.event_source.stop()
            
            status_text.set_text(f"GOAL REACHED at step {step_count}!\n"
                               f"Total cost: {total_cost:.2f}\n"
                               f"3D error: {distance_to_goal:.2f}m")
            
            return ([quad_marker_3d, traj_line_3d, roll_line, pitch_line, yaw_line, 
                    goal_dist_line, quad_marker_2d, traj_line_2d, quad_height_line, 
                    status_text] + obs_markers_3d + obs_markers_2d + 
                    obs_traj_lines + height_lines)
        
        # Update obstacle positions
        for i, obs in enumerate(obstacles):
            prev_pos = obs.get_position().copy()
            # Pass goal_z to ensure obstacles maintain the same altitude as the goal
            obs.update_position(mpc.dt, use_gmm=True, maintain_z=True, goal_z=goal[2])
            
            # Record obstacle trajectory
            obstacle_trajectories[i].append(obs.get_position().copy())
        
        # Generate control sequence using MPC - pass the full 3D goal
        control_sequence = mpc.optimize_trajectory(current_state, goal, obstacles)
        
        if control_sequence is not None:
            control_cost = np.sum(np.linalg.norm(control_sequence[0])**2)
            
            quadrotor.update_state(control_sequence[0], mpc.dt)
            
            new_state = quadrotor.get_state()
            state_cost = np.linalg.norm(new_state[:3] - goal)**2
            
            # More reasonable cost calculation with capping
            step_cost = state_cost + 0.1 * control_cost
            step_cost = min(step_cost, 100)  # Cap to prevent huge costs
            total_cost += step_cost
        else:
            print("Warning: No valid control found!")
        
        # Update 3D visualization
        current_state = quadrotor.get_state()
        position = current_state[:3]
        orientation = current_state[6:9]  # phi, theta, psi
        
        quad_marker_3d.set_data([position[0]], [position[1]])
        quad_marker_3d.set_3d_properties([position[2]])
        
        # Line from quadrotor to goal to show true distance
        goal_dist_line.set_data([position[0], goal[0]], [position[1], goal[1]])
        goal_dist_line.set_3d_properties([position[2], goal[2]])
        
        # Draw orientation lines
        roll_dir, pitch_dir, yaw_dir = draw_orientation(position, orientation)
        
        roll_line.set_data([position[0], position[0] + roll_dir[0]], 
                           [position[1], position[1] + roll_dir[1]])
        roll_line.set_3d_properties([position[2], position[2] + roll_dir[2]])
        
        pitch_line.set_data([position[0], position[0] + pitch_dir[0]], 
                            [position[1], position[1] + pitch_dir[1]])
        pitch_line.set_3d_properties([position[2], position[2] + pitch_dir[2]])
        
        yaw_line.set_data([position[0], position[0] + yaw_dir[0]], 
                          [position[1], position[1] + yaw_dir[1]])
        yaw_line.set_3d_properties([position[2], position[2] + yaw_dir[2]])
        
        # Update 3D trajectory with better visibility
        trajectory = np.array(actual_trajectory)
        if len(trajectory) > 0:
            traj_line_3d.set_data(trajectory[:, 0], trajectory[:, 1])
            traj_line_3d.set_3d_properties(trajectory[:, 2])
            
            # Periodically update the 3D view to follow the quadrotor
            if step_count % 20 == 0:
                # Adjust view to keep quadrotor in frame
                ax1.view_init(elev=30, azim=45)
                
        # Update top-down view (now also 3D)
        quad_marker_2d.set_data([position[0]], [position[1]])
        quad_marker_2d.set_3d_properties([position[2]])
        
        # Add height line for quadrotor in top-down view
        quad_height_line.set_data([position[0], position[0]], [position[1], position[1]])
        quad_height_line.set_3d_properties([0, position[2]])
        
        if len(trajectory) > 0:
            traj_line_2d.set_data(trajectory[:, 0], trajectory[:, 1])
            traj_line_2d.set_3d_properties(trajectory[:, 2])
        
        # Update obstacle trajectories
        for i, obs_traj in enumerate(obstacle_trajectories):
            traj_arr = np.array(obs_traj)
            obs_traj_lines[i].set_data(traj_arr[:, 0], traj_arr[:, 1])
            obs_traj_lines[i].set_3d_properties(traj_arr[:, 2])
        
        # Update obstacle visualization
        for i, obs in enumerate(obstacles):
            obs_pos = obs.get_position()
            
            # Update markers in main 3D view
            obs_markers_3d[i].set_data([obs_pos[0]], [obs_pos[1]])
            obs_markers_3d[i].set_3d_properties([obs_pos[2]])
            
            # Update markers in top-down view
            obs_markers_2d[i].set_data([obs_pos[0]], [obs_pos[1]])
            obs_markers_2d[i].set_3d_properties([obs_pos[2]])
            
            # Update height lines in top-down view
            height_lines[i].set_data([obs_pos[0], obs_pos[0]], [obs_pos[1], obs_pos[1]])
            height_lines[i].set_3d_properties([0, obs_pos[2]])
        
        return ([quad_marker_3d, traj_line_3d, roll_line, pitch_line, yaw_line, 
                goal_dist_line, quad_marker_2d, traj_line_2d, quad_height_line, 
                status_text] + obs_markers_3d + obs_markers_2d + 
                obs_traj_lines + height_lines)
    
    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=300,
        init_func=init,
        blit=True,
        interval=100
    )
    
    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    # Add title with paper reference
    fig.text(0.5, 0.01, "3D Implementation of 'Online-Learning-Based Distributionally Robust Motion Control'", 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Either save animation or show interactive plot
    # ani.save('real_quadrotor_3d_visualization.mp4', writer='ffmpeg', fps=10, dpi=200)
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
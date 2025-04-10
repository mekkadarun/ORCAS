import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from motion_models.quadrotor import Quadrotor
from motion_models.gmm_obstacle import GMMObstacle
from control.gmm_mpc import CVaRGMMMPC

def draw_confidence_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(*mean)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_risk_heatmap(ax, obstacles, cvar_controller, x_range, y_range, resolution=0.2):
    risk_map, X, Y = cvar_controller.cvar_avoidance.generate_risk_map(
        x_range, y_range, resolution, obstacles
    )
    
    heatmap = ax.pcolormesh(X, Y, risk_map, cmap='Reds', alpha=0.4, 
                          vmin=0, vmax=1, shading='auto')
    
    return heatmap

# Initialize components
quadrotor = Quadrotor(np.array([0.0, 0.0, 0.0, 0.0]))
obstacles = [
    GMMObstacle((2.0, 2.0), radius=0.3, n_components=2),
    GMMObstacle((4.0, 4.0), radius=0.3, n_components=2),
    GMMObstacle((6.0, 1.0), radius=0.3, n_components=2)
]
goal = np.array([8.0, 8.0])

# Initialize CVaR-based MPC controller
mpc = CVaRGMMMPC(horizon=8, dt=0.1, quad_radius=0.3, confidence_level=0.95)

# Tracking variables
step_count = 0
total_cost = 0.0
actual_trajectory = []

# Pre-train GMM models with distinct movement patterns
movement_patterns = [
    ([0.1, 0.2], [[0.02, 0.005], [0.005, 0.02]]),
    ([0.2, -0.1], [[0.03, -0.01], [-0.01, 0.03]]),
    ([0.05, 0.15], [[0.04, 0.02], [0.02, 0.04]])
]

# Generate initial movement data
for i, obs in enumerate(obstacles):
    pattern_idx = i % len(movement_patterns)
    mean, cov = movement_patterns[pattern_idx]
    
    for _ in range(30):
        movement = np.random.multivariate_normal(mean=mean, cov=cov)
        obs.movement_history.append(movement)
    
    for _ in range(5):
        outlier = np.random.multivariate_normal(
            mean=[-m for m in mean],
            cov=[[0.1, 0], [0, 0.1]]
        )
        obs.movement_history.append(outlier)
    
    obs._fit_gmm()

# Create figure for visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_title("CVaR-based ORCAS with GMM Uncertainty", fontsize=16)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Plot static elements
goal_marker = ax.scatter(*goal, s=200, c='green', marker='*', label='Goal')

# Dynamic elements to be updated
quad_marker, = ax.plot([], [], 'o', color='blue', markersize=10, label='Quadrotor')
traj_line, = ax.plot([], [], '--', color='blue', label='Trajectory')

# Status text display
status_text = ax.text(
    0.02, 0.98, '', 
    transform=ax.transAxes, 
    verticalalignment='top',
    fontsize=12,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
)

# Draw initial risk heatmap
risk_heatmap = draw_risk_heatmap(ax, obstacles, mpc, (-1, 10), (-1, 10), resolution=0.2)

# Obstacle markers and uncertainty ellipses
obs_markers = []
ellipse_patches = []

# Initialize visualization
for obs in obstacles:
    circle = plt.Circle(obs.get_position(), obs.radius, color='red', alpha=0.7)
    ax.add_patch(circle)
    obs_markers.append(circle)
    ellipse_patches.append([])

def init():
    quad_marker.set_data([], [])
    traj_line.set_data([], [])
    status_text.set_text('Starting simulation...')
    return [quad_marker, traj_line, status_text, risk_heatmap] + obs_markers

def update(frame):
    global step_count, total_cost, risk_heatmap
    step_count += 1
    
    current_state = quadrotor.get_state()
    actual_trajectory.append(current_state[:2].copy())
    
    distance_to_goal = np.linalg.norm(current_state[:2] - goal)
    
    status_text.set_text(f"Step: {step_count} | Dist to goal: {distance_to_goal:.2f}m | Cost: {total_cost:.2f}")
    
    if distance_to_goal < 0.2:
        print(f"Goal reached at step {step_count}!")
        print(f"Total cost: {total_cost:.2f}")
        print(f"Final position: {current_state[:2]}")
        ani.event_source.stop()
        
        status_text.set_text(f"GOAL REACHED at step {step_count}!\nTotal cost: {total_cost:.2f}")
        
        return [quad_marker, traj_line, status_text, risk_heatmap] + obs_markers + get_ellipses()
    
    for obs in obstacles:
        obs.update_position(mpc.dt, use_gmm=True)
    
    if step_count % 10 == 0:
        risk_heatmap.remove()
        risk_heatmap = draw_risk_heatmap(ax, obstacles, mpc, (-1, 10), (-1, 10), resolution=0.2)
    
    control_sequence = mpc.optimize_trajectory(current_state, goal, obstacles)
    
    if control_sequence is not None:
        control_cost = np.sum(np.linalg.norm(control_sequence[0])**2)
        
        quadrotor.update_state(control_sequence[0], mpc.dt)
        
        new_state = quadrotor.get_state()
        state_cost = np.linalg.norm(new_state[:2] - goal)**2
        
        step_cost = state_cost + 0.1 * control_cost
        total_cost += step_cost
    else:
        print("Warning: No valid control found!")
    
    quad_marker.set_data([current_state[0]], [current_state[1]])
    
    trajectory = np.array(actual_trajectory)
    if len(trajectory) > 0:
        traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
    
    for i, (obs, marker) in enumerate(zip(obstacles, obs_markers)):
        marker.center = obs.get_position()
        
        for patch in ellipse_patches[i]:
            patch.remove()
        ellipse_patches[i] = []
        
        params = obs.get_constraint_parameters()
        if 'uncertainty' in params and len(params['uncertainty']) > 0:
            for t in range(min(3, len(params['uncertainty']))):
                uncertainty = params['uncertainty'][t]
                
                for j, (mean, cov, weight) in enumerate(zip(
                    uncertainty['means'], 
                    uncertainty['covariances'],
                    uncertainty['weights']
                )):
                    if weight < 0.05:
                        continue
                        
                    alpha = weight * (0.6 - t * 0.2)
                    
                    ellipse = draw_confidence_ellipse(
                        ax, mean, cov, 
                        n_std=1.0,
                        edgecolor='orange', 
                        facecolor='yellow',
                        alpha=alpha,
                        linewidth=1
                    )
                    ellipse_patches[i].append(ellipse)
    
    return [quad_marker, traj_line, status_text, risk_heatmap] + obs_markers + get_ellipses()

def get_ellipses():
    return [p for patches in ellipse_patches for p in patches]

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=300,
    init_func=init,
    blit=True,
    interval=100
)

if __name__ == "__main__":
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
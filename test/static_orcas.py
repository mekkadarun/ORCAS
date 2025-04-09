import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from motion_models.quadrotor import Quadrotor
from motion_models.static_obstacle import Obstacle
from control.static_mpc import MPC

# Initialize components
quadrotor = Quadrotor(np.array([0, 0, 0, 0]))
obstacles = [
    Obstacle((2, 2), radius=0.1),
    Obstacle((3, 4), radius=0.15),
    Obstacle((3, 3), radius=0.35),
    Obstacle((5, 3), radius=0.05)
]
goal = np.array([5, 5])
# Initialize MPC with quadrotor radius parameter
mpc = MPC(horizon=8, dt=0.1, quad_radius=0.3)

# Simulation parameters
actual_trajectory = []
control_sequence = None

# Create figure for animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_title("ORCAS - STATIC OBSTACLES", fontsize=16)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Plot static elements: obstacles and goal
for obs in obstacles:
    circle = plt.Circle(obs.get_position(), obs.radius, color='red', alpha=0.7)
    ax.add_patch(circle)
goal_marker = ax.scatter(*goal, s=200, c='green', marker='*', label='Goal')

# Dynamic elements: quadrotor and trajectory
quad_marker, = ax.plot([], [], 'o', color='orange', markersize=10, label='Quadrotor')
trajectory_line, = ax.plot([], [], '--', color='blue', label='Trajectory')

# Initialize animation function
def init():
    quad_marker.set_data([], [])
    trajectory_line.set_data([], [])
    return quad_marker, trajectory_line

# Animation update function
def update(frame):
    global control_sequence
    
    # Get current state and add to trajectory
    current_state = quadrotor.get_state()
    actual_trajectory.append(current_state[:2].copy())
    
    # Check if goal is reached
    if np.linalg.norm(current_state[:2] - goal) < 0.2:
        print(f"Goal reached at frame {frame}!")
        ani.event_source.stop()
        return quad_marker, trajectory_line
    
    # Optimize trajectory and apply first control input
    control_sequence = mpc.optimize_trajectory(current_state, goal, obstacles)
    quadrotor.update_state(control_sequence[0], mpc.dt)
    
    # Update dynamic elements in the plot
    quad_marker.set_data([current_state[0]], [current_state[1]])

    trajectory = np.array(actual_trajectory)
    if len(trajectory) > 0:
        trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
    else:
        trajectory_line.set_data([], [])
    
    return quad_marker, trajectory_line

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=100,
    init_func=init,
    blit=True,
    interval=100  # Frame interval in milliseconds (100ms → 10 FPS)
)

# Show animation
plt.legend()
plt.show()

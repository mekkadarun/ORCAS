import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from motion_models.quadrotor import Quadrotor
from motion_models.dynamic_obstacle import DynamicObstacle
from control.dynamic_mpc import DynamicMPC

# Initialize components with explicit float values
quadrotor = Quadrotor(np.array([0.0, 0.0, 0.0, 0.0]))
obstacles = [
    DynamicObstacle(position=(2.0, 2.0), velocity=(0.5, 0.3), radius=0.5),
    DynamicObstacle(position=(5.0, 3.0), velocity=(-0.2, 0.5), radius=0.5),
    DynamicObstacle(position=(5.0, 3.0), velocity=(-0.1, 1), radius=0.5),
    DynamicObstacle(position=(5.0, 3.0), velocity=(-0.2, 0.3), radius=0.5),
    DynamicObstacle(position=(5.0, 3.0), velocity=(-0.5, 0.1), radius=0.5)
]
goal = np.array([8.0, 8.0])
mpc = DynamicMPC(horizon=8, dt=0.1, quad_radius=0.3)

# Setup visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_title("Dynamic Obstacle Avoidance", fontsize=16)

# Plot elements
goal_marker = ax.scatter(*goal, s=200, c='g', marker='*')
quad_marker, = ax.plot([], [], 'o', color='orange', markersize=10)
traj_line, = ax.plot([], [], 'b--')
obs_markers = [ax.plot([], [], 's', color='r', markersize=10)[0] for _ in obstacles]

def init():
    quad_marker.set_data([], [])
    traj_line.set_data([], [])
    for m in obs_markers: m.set_data([], [])
    return [quad_marker, traj_line] + obs_markers

def update(frame):
    # Update obstacle positions
    for obs in obstacles:
        obs.update_position(mpc.dt)
    
    # MPC optimization
    state = quadrotor.get_state()
    ctrl = mpc.optimize_trajectory(state, goal, obstacles)
    
    if ctrl is not None:
        quadrotor.update_state(ctrl[0], mpc.dt)
    
    # Check goal proximity
    current_pos = quadrotor.get_position()
    distance_to_goal = np.linalg.norm(current_pos - goal)
    if distance_to_goal < 0.2:
        print(f"Goal reached at step {frame}!")
        ani.event_source.stop()
        return [quad_marker, traj_line] + obs_markers
    
    # Update visualization
    quad_marker.set_data([current_pos[0]], [current_pos[1]])
    traj_line.set_data(*zip(*quadrotor.trajectory))
    
    for i, obs in enumerate(obstacles):
        obs_markers[i].set_data([obs.position[0]], [obs.position[1]])
    
    return [quad_marker, traj_line] + obs_markers

# Run simulation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100)
plt.show()

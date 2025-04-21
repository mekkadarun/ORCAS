import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as colors

from motion_models.quadrotor import Quadrotor
from motion_models.gmm_obstacle import GMMObstacle
from control.gmm_mpc import CVaRGMMMPC

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
    """Run the simulation and create an animation."""
    # Initialize components
    quadrotor = Quadrotor(np.array([0.0, 0.0, 0.0, 0.0]))
    obstacles = [
        GMMObstacle((2.0, 2.0), radius=0.3, n_components=2),
        GMMObstacle((4.0, 4.0), radius=0.4, n_components=2),
        GMMObstacle((2.70, 1.0), radius=0.3, n_components=2)
    ]
    goal = np.array([7.0, 7.0])

    # Initialize CVaR-based MPC controller
    mpc = CVaRGMMMPC(horizon=8, dt=0.1, quad_radius=0.3, confidence_level=0.95)

    # Tracking variables
    step_count = 0
    total_cost = 0.0
    actual_trajectory = []
    obstacle_trajectories = [[] for _ in obstacles]

    # Pre-train GMM models with distinct movement patterns
    movement_patterns = [
        ([0.1, 0.2], [[0.02, 0.005], [0.005, 0.02]]),
        ([0.1, -0.1], [[0.03, -0.01], [-0.01, 0.03]]),
        ([0.1, 0.15], [[0.04, 0.02], [0.02, 0.04]])
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
                cov=[[0.1, 0], [0, 0.1]]
            )
            obs.movement_history.append(outlier)
            obs.ambiguity_set.add_movement_data(outlier)
        
        # Initialize GMM and ambiguity set
        obs._fit_gmm()
        obs.ambiguity_set.update_mixture_model()
        
        # Record initial position
        obstacle_trajectories[i].append(obs.get_position().copy())

    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.set_title("Online-Learning-Based Distributionally Robust Motion Control", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    # Plot static elements
    goal_marker = ax.scatter(*goal, s=200, c='green', marker='*', label='Goal', zorder=10)

    # Dynamic elements to be updated
    quad_marker, = ax.plot([], [], 'o', color='blue', markersize=10, label='Quadrotor', zorder=10)
    traj_line, = ax.plot([], [], '--', color='blue', label='Trajectory', zorder=5)

    # Obstacle trajectories
    obs_traj_lines = []
    for i in range(len(obstacles)):
        line, = ax.plot([], [], ':', color=f'C{i+1}', label=f'Obstacle {i+1} Trajectory', alpha=0.6, zorder=4)
        obs_traj_lines.append(line)

    # Status text display
    status_text = ax.text(
        0.02, 0.98, '', 
        transform=ax.transAxes, 
        verticalalignment='top',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        zorder=20
    )

    # Draw initial risk heatmap
    risk_heatmap = draw_risk_heatmap(ax, obstacles, mpc, (-1, 10), (-1, 10), resolution=0.2)

    # Obstacle markers and uncertainty ellipses
    obs_markers = []
    ellipse_patches = []

    # Initialize visualization
    for i, obs in enumerate(obstacles):
        circle = plt.Circle(obs.get_position(), obs.radius, 
                          color=f'C{i+1}', alpha=0.7, label=f'Obstacle {i+1}', zorder=6)
        ax.add_patch(circle)
        obs_markers.append(circle)
        ellipse_patches.append([])
    
    # Risk map object
    risk_map_object = [risk_heatmap]

    def init():
        """Initialize animation elements."""
        quad_marker.set_data([], [])
        traj_line.set_data([], [])
        for line in obs_traj_lines:
            line.set_data([], [])
        status_text.set_text('Starting simulation...')
        return [quad_marker, traj_line, status_text] + risk_map_object + obs_markers + obs_traj_lines + get_ellipses()

    def update(frame):
        """Update animation for each frame."""
        nonlocal step_count, total_cost
        step_count += 1
        
        current_state = quadrotor.get_state()
        actual_trajectory.append(current_state[:2].copy())
        
        distance_to_goal = np.linalg.norm(current_state[:2] - goal)
        
        status_text.set_text(f"Step: {step_count} | Distance to goal: {distance_to_goal:.2f}m | Cost: {total_cost:.2f}")
        
        # Check if goal is reached
        if distance_to_goal < 0.4:
            print(f"Goal reached at step {step_count}!")
            print(f"Total cost: {total_cost:.2f}")
            print(f"Final position: {current_state[:2]}")
            ani.event_source.stop()
            
            status_text.set_text(f"GOAL REACHED at step {step_count}!\nTotal cost: {total_cost:.2f}")
            
            return [quad_marker, traj_line, status_text] + risk_map_object + obs_markers + obs_traj_lines + get_ellipses()
        
        # Update obstacle positions
        for i, obs in enumerate(obstacles):
            prev_pos = obs.get_position().copy()
            obs.update_position(mpc.dt, use_gmm=True)
            
            # Record obstacle trajectory
            obstacle_trajectories[i].append(obs.get_position().copy())
        
        # Periodically update risk heatmap
        if step_count % 10 == 0:
            risk_map_object[0].remove()
            risk_map_object[0] = draw_risk_heatmap(ax, obstacles, mpc, (-1, 10), (-1, 10), resolution=0.2)
        
        # Generate control sequence using MPC
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
        
        # Update visualization
        quad_marker.set_data([current_state[0]], [current_state[1]])
        
        trajectory = np.array(actual_trajectory)
        if len(trajectory) > 0:
            traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
        
        # Update obstacle trajectories
        for i, obs_traj in enumerate(obstacle_trajectories):
            traj_arr = np.array(obs_traj)
            obs_traj_lines[i].set_data(traj_arr[:, 0], traj_arr[:, 1])
        
        # Update obstacle visualization
        for i, (obs, marker) in enumerate(zip(obstacles, obs_markers)):
            marker.center = obs.get_position()
            
            # Remove previous ellipses
            for patch in ellipse_patches[i]:
                patch.remove()
            ellipse_patches[i] = []
            
            # Get uncertainty parameters for visualization
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
                            edgecolor=f'C{i+1}',
                            facecolor=f'C{i+1}',
                            alpha=alpha,
                            linewidth=1,
                            zorder=3
                        )
                        ellipse_patches[i].append(ellipse)
        
        return [quad_marker, traj_line, status_text] + risk_map_object + obs_markers + obs_traj_lines + get_ellipses()

    def get_ellipses():
        """Helper function to get all ellipse patches for animation."""
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
    
    # Add legend in a good location
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Add text box with explanation
    textstr = '\n'.join((
        'Online Learning:',
        '- Obstacles learn movement patterns',
        '- Ambiguity sets are updated online',
        '- Ellipses show 1σ uncertainty bounds',
        '- Heatmap shows combined risk',
        '',
        'Distributionally Robust Control:',
        '- Safe direction approach',
        '- CVaR-based risk assessment',
        '- Takes worst-case distributions into account',
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.7, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add title with paper reference
    fig.text(0.5, 0.01, "Implementation of 'Online-Learning-Based Distributionally Robust Motion Control'", 
             ha='center', fontsize=12, style='italic')
    
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Either save animation or show interactive plot
    # ani.save('gmm_orcas_visualization.mp4', writer='ffmpeg', fps=10, dpi=200)
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
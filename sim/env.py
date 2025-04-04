import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional, Union, Any

from motion_models.quadrotor import QuadrotorModel
from motion_models.obstacle import ObstacleSet, GaussianObstacle, GMMMixingObstacle, TimeVaryingGaussianObstacle
from learning.gmm import OnlineDistributionLearner
from control.mpc import MPC, DistributionallyRobustMPC


class QuadrotorEnvironment:
    """
    Simulation environment for the quadrotor with moving obstacles.
    Integrates the quadrotor model, obstacles, and controllers.
    """
    
    def __init__(self, dt: float = 0.1, enable_learning: bool = True):
        """
        Initialize the environment.
        
        Args:
            dt: Time step of the simulation
            enable_learning: Whether to enable online learning
        """
        # Time settings
        self.dt = dt
        self.time = 0.0
        
        # Quadrotor model
        self.quadrotor = QuadrotorModel(dt=dt)
        self.state = np.zeros(12)
        self.state[2] = 2.0  # Start at 2m height
        
        # Target
        self.target = np.array([15.0, 15.0, 5.0])
        
        # Trajectories
        self.state_trajectory = [self.state.copy()]
        
        # Initialize obstacles
        self.obstacle_set = ObstacleSet()
        self._setup_obstacles()
        
        # Online learning
        self.enable_learning = enable_learning
        self.learner = OnlineDistributionLearner()
        
        # For each obstacle, create an ambiguity set
        if self.enable_learning:
            for i, obstacle in enumerate(self.obstacle_set.get_obstacles()):
                self.learner.create_ambiguity_set(i, dimension=3)
        
        # Reference trajectory (simple straight line to target)
        self.reference_trajectory = self._generate_reference_trajectory()
        
        # Controller (will be set later)
        self.controller = None
        
    def _setup_obstacles(self):
        """
        Set up the obstacles for the environment.
        One multimodal obstacle and one time-varying obstacle.
        """
        # First obstacle: Multimodal Gaussian
        pos1 = np.array([7.0, 5.0, 3.0])
        radius1 = 0.8
        
        # Define GMM parameters for the first obstacle
        means = [np.array([0.1, 0.05, 0.0]), np.array([-0.05, 0.1, 0.0])]
        covariances = [np.diag([0.01, 0.01, 0.01]), np.diag([0.01, 0.01, 0.01])]
        weights = [0.6, 0.4]
        
        obstacle1 = GMMMixingObstacle(pos1, radius1, means, covariances, weights)
        self.obstacle_set.add_obstacle(obstacle1)
        
        # Second obstacle: Time-varying Gaussian
        pos2 = np.array([12.0, 10.0, 5.0])
        radius2 = 0.8
        
        # Time-varying parameters
        initial_mean = np.array([0.05, 0.05, 0.0])
        initial_cov = np.diag([0.01, 0.01, 0.01])
        variance_growth_rate = 0.01  # Rate of growth for variance
        
        obstacle2 = TimeVaryingGaussianObstacle(pos2, radius2, initial_mean, initial_cov, variance_growth_rate)
        self.obstacle_set.add_obstacle(obstacle2)
        
    def _generate_reference_trajectory(self, num_points: int = 100) -> np.ndarray:
        """
        Generate a simple straight-line reference trajectory to target.
        
        Args:
            num_points: Number of points in the trajectory
            
        Returns:
            Reference trajectory array
        """
        start = self.state[:3].copy()
        target = self.target.copy()
        
        # Create a straight line trajectory
        trajectory = []
        for i in range(num_points):
            alpha = i / (num_points - 1)
            point = start * (1 - alpha) + target * alpha
            trajectory.append(point)
            
        return np.array(trajectory)
    
    def set_controller(self, controller: Union[MPC, DistributionallyRobustMPC]):
        """
        Set the controller for the quadrotor.
        
        Args:
            controller: MPC controller
        """
        self.controller = controller
        
    def step(self) -> Tuple[bool, bool]:
        """
        Execute one time step of the simulation.
        
        Returns:
            Tuple of (success, collision)
        """
        if self.controller is None:
            raise ValueError("Controller not set")
        
        # Move obstacles
        self.obstacle_set.update_all(self.dt)
        
        # Collect movement data for learning
        if self.enable_learning:
            for i, obstacle in enumerate(self.obstacle_set.get_obstacles()):
                if hasattr(obstacle, 'movements') and len(obstacle.movements) > 0:
                    last_movement = obstacle.movements[-1]
                    self.learner.add_movement_data(i, last_movement)
            
            # Update ambiguity sets
            self.learner.update_all()
        
        # Prepare obstacle information for controller
        obstacles_info = []
        for i, obstacle in enumerate(self.obstacle_set.get_obstacles()):
            if self.enable_learning:
                ambiguity_set = self.learner.get_ambiguity_set(i)
                if ambiguity_set:
                    components = ambiguity_set.get_components()
                else:
                    # Fallback if ambiguity set not ready
                    components = [{'weight': 1.0, 'mean': np.zeros(3), 'covariance': np.eye(3)}]
            else:
                # If learning disabled, use simple components
                components = [{'weight': 1.0, 'mean': np.zeros(3), 'covariance': np.eye(3)}]
                
            obstacles_info.append({
                'position': obstacle.position,
                'safety_radius': obstacle.radius + 0.5,  # Add safety margin
                'ambiguity_components': components
            })
        
        # Choose relevant part of reference trajectory
        reference = self.reference_trajectory[min(len(self.state_trajectory)-1, 
                                               len(self.reference_trajectory)-10):
                                            min(len(self.state_trajectory)-1+10, 
                                               len(self.reference_trajectory))]
        
        if len(reference) < 10:
            # Pad with last point if needed
            padding = [reference[-1] for _ in range(10 - len(reference))]
            reference = np.vstack([reference, padding])
        
        # Get control input from the controller
        if isinstance(self.controller, DistributionallyRobustMPC):
            control = self.controller.solve_with_obstacles(
                self.state, reference, obstacles_info
            )
        else:
            control = self.controller.solve(
                self.state, reference
            )
            
        if control is None:
            # Fall back to a simple controller if MPC fails
            direction = self.target - self.state[:3]
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            control = np.array([direction[0], direction[1], direction[2]]) * 2.0
            
        # Apply control input
        self.state = self.quadrotor.forward_dynamics(self.state, control)
        
        # Record state
        self.state_trajectory.append(self.state.copy())
        
        # Update time
        self.time += self.dt
        
        # Check for collisions
        collision = self._check_collision()
        
        # Check if reached target
        success = np.linalg.norm(self.state[:3] - self.target) < 1.0
        done = collision or success
        
        return success, collision
    
    def _check_collision(self) -> bool:
        """
        Check if the quadrotor collides with any obstacle.
        
        Returns:
            True if collision detected, False otherwise
        """
        return self.obstacle_set.check_collision(self.state[:3], safety_margin=0.5)
    
    def run_simulation(self, max_steps: int = 200) -> Dict[str, Any]:
        """
        Run the simulation for a maximum number of steps.
        
        Args:
            max_steps: Maximum number of steps to run
            
        Returns:
            Dictionary with simulation results
        """
        steps = 0
        success = False
        collision = False
        
        for _ in range(max_steps):
            success, collision = self.step()
            steps += 1
            
            if success:
                print(f"Target reached at step {steps}")
                break
            
            if collision:
                print(f"Collision occurred at step {steps}")
                break
        
        return {
            'steps': steps,
            'success': success,
            'collision': collision,
            'state_trajectory': np.array(self.state_trajectory),
            'obstacle_trajectories': [np.array(obs.trajectory) for obs in self.obstacle_set.get_obstacles()]
        }
    
    def visualize(self, block: bool = True):
        """
        Visualize the simulation results.
        
        Args:
            block: Whether to block execution until the figure is closed
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot quadrotor trajectory
        trajectory = np.array(self.state_trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', lw=2, label='Quadrotor')
        
        # Plot start and end positions
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='blue', s=100, marker='o')
        
        # Plot target
        ax.scatter(self.target[0], self.target[1], self.target[2], color='red', s=100, marker='*', label='Target')
        
        # Plot reference trajectory
        ref_traj = self.reference_trajectory
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 'k--', alpha=0.5, label='Reference')
        
        # Plot obstacles and their trajectories
        obstacles = self.obstacle_set.get_obstacles()
        colors = ['red', 'green']
        
        for i, obstacle in enumerate(obstacles):
            obs_traj = np.array(obstacle.trajectory)
            ax.plot(obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2], color=colors[i], alpha=0.5, label=f'Obstacle {i+1}')
            
            # Draw final obstacle position as sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = obstacle.position[0] + obstacle.radius * np.cos(u) * np.sin(v)
            y = obstacle.position[1] + obstacle.radius * np.sin(u) * np.sin(v)
            z = obstacle.position[2] + obstacle.radius * np.cos(v)
            ax.plot_surface(x, y, z, color=colors[i], alpha=0.3)
        
        # Set labels and limits
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Auto-adjust limits with some padding
        x_data = np.concatenate([trajectory[:, 0], [self.target[0]]] + [obs.position[0] for obs in obstacles])
        y_data = np.concatenate([trajectory[:, 1], [self.target[1]]] + [obs.position[1] for obs in obstacles])
        z_data = np.concatenate([trajectory[:, 2], [self.target[2]]] + [obs.position[2] for obs in obstacles])
        
        x_min, x_max = min(x_data) - 1, max(x_data) + 1
        y_min, y_max = min(y_data) - 1, max(y_data) + 1
        z_min, z_max = min(z_data) - 1, max(z_data) + 1
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Set title based on simulation outcome
        if self._check_collision():
            plt.title('Simulation Result: Collision Occurred', color='red')
        elif np.linalg.norm(self.state[:3] - self.target) < 1.0:
            plt.title('Simulation Result: Target Reached Successfully', color='green')
        else:
            plt.title('Simulation Result: In Progress')
        
        ax.legend()
        plt.tight_layout()
        plt.show(block=block)
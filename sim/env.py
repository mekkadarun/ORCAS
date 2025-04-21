import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, obstacles, goal):
        self.obstacles = obstacles
        self.goal = goal
        
    def plot_environment(self, quadrotor_state, trajectory):
        """Plot the environment, trajectory and current quadrotor position"""
        plt.figure(figsize=(10, 10))
        
        # Plot obstacles
        for obs in self.obstacles:
            pos = obs.get_position()
            circle = plt.Circle(pos, obs.radius, color='red', alpha=0.7)
            plt.gca().add_patch(circle)
            
        # Plot goal
        plt.scatter(*self.goal, s=200, c='green', marker='*', label='Goal')
        
        # Plot trajectory
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b--', linewidth=2, label='Planned Path')
        
        # Plot quadrotor
        plt.scatter(quadrotor_state[0], quadrotor_state[1], 
                  s=100, c='orange', marker='^', label='Quadrotor')
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Quadrotor Trajectory with Obstacle Avoidance')
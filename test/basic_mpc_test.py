# obstacle_avoidance_test.py
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the path to import modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_models.quadrotor import QuadrotorModel
from motion_models.obstacle import StaticObstacle
from control.mpc import MPC, DistributionallyRobustMPC
from sim.env import QuadrotorEnvironment

def test_with_obstacle():
    """Test with simplified dynamics and a single obstacle."""
    # Create environment
    env = QuadrotorEnvironment(dt=0.1, enable_learning=False)
    env.obstacle_set.obstacles = []
    
    # Add a single static obstacle in the path
    obstacle_position = np.array([2.5, 2.5, 3.0])  # In the middle of the path
    obstacle_radius = 0.5
    static_obstacle = StaticObstacle(obstacle_position, obstacle_radius)
    env.obstacle_set.add_obstacle(static_obstacle)
    
    # Define target
    env.target = np.array([5.0, 5.0, 3.0])
    
    # Generate a reference trajectory around the obstacle
    # We'll create a curved path instead of a straight line
    start = env.state[:3].copy()
    target = env.target.copy()
    obstacle_pos = obstacle_position
    
    # Vector from start to target
    direct_vector = target - start
    direct_distance = np.linalg.norm(direct_vector)
    
    # Vector perpendicular to the direct path (in the xy plane)
    perp_vector = np.array([-direct_vector[1], direct_vector[0], 0])
    perp_vector = perp_vector / np.linalg.norm(perp_vector)
    
    # Create a curved path to avoid the obstacle
    trajectory = []
    num_points = 50
    for i in range(num_points):
        t = i / (num_points - 1)  # Parameter from 0 to 1
        
        # Apply a sine curve to create a detour
        detour_amplitude = 1.5  # How far to detour
        detour_factor = np.sin(t * np.pi) * detour_amplitude
        
        # Calculate point
        point = start + t * direct_vector + perp_vector * detour_factor
        trajectory.append(point)
    
    env.reference_trajectory = np.array(trajectory)
    
    # Override the forward_dynamics method with simplified dynamics
    original_dynamics = env.quadrotor.forward_dynamics
    
    # Add the simplified_dynamics method to the quadrotor model
    def simplified_dynamics(self, state, control):
        """A simplified version of the dynamics that's easier to control."""
        x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot = state
        u1, u2, u3, u4 = control
        
        # Direct control of velocities
        x_ddot = u2  # Roll control affects x acceleration directly
        y_ddot = u3  # Pitch control affects y acceleration directly
        z_ddot = (1/self.mc) * u1 - self.g  # Thrust affects z, countering gravity
        
        # Simple angular dynamics
        phi_ddot = u2 * 5  # Scale up to make angles change more quickly
        theta_ddot = u3 * 5
        psi_ddot = u4
        
        # Euler integration
        vx_next = vx + x_ddot * self.dt
        vy_next = vy + y_ddot * self.dt
        vz_next = vz + z_ddot * self.dt
        
        x_next = x + vx * self.dt
        y_next = y + vy * self.dt
        z_next = z + vz * self.dt
        
        phi_dot_next = phi_dot + phi_ddot * self.dt
        theta_dot_next = theta_dot + theta_ddot * self.dt
        psi_dot_next = psi_dot + psi_ddot * self.dt
        
        phi_next = phi + phi_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        psi_next = psi + psi_dot * self.dt
        
        next_state = np.array([
            x_next, y_next, z_next, vx_next, vy_next, vz_next,
            phi_next, theta_next, psi_next, phi_dot_next, theta_dot_next, psi_dot_next
        ])
        
        return next_state
    
    # Add the method to the quadrotor model
    QuadrotorModel.simplified_dynamics = simplified_dynamics
    
    # Set forward_dynamics to the simplified version
    env.quadrotor.forward_dynamics = env.quadrotor.simplified_dynamics
    
    # Create a state-space model for the simplified dynamics
    dt = 0.1
    
    # A matrix - simple double integrator with identity for angles
    A = np.eye(12)
    # Position updates with velocity
    A[0, 3] = A[1, 4] = A[2, 5] = dt
    # Angles update with angular velocity
    A[6, 9] = A[7, 10] = A[8, 11] = dt
    
    # B matrix - direct control
    B = np.zeros((12, 4))
    B[5, 0] = dt / env.quadrotor.mc  # Thrust affects z acceleration
    B[3, 1] = dt  # u2 affects x acceleration
    B[4, 2] = dt  # u3 affects y acceleration
    B[9, 1] = dt * 5  # u2 affects phi_dot
    B[10, 2] = dt * 5  # u3 affects theta_dot
    B[11, 3] = dt  # u4 affects psi_dot
    
    # Output matrix - just positions
    C = np.zeros((3, 12))
    C[0, 0] = C[1, 1] = C[2, 2] = 1
    
    # Cost matrices
    Q = np.diag([10.0, 10.0, 10.0])
    R = np.diag([0.1, 1.0, 1.0, 0.1])
    
    # Control constraints
    control_constraints = {
        'lower': np.array([env.quadrotor.mc * env.quadrotor.g * 0.5, -2.0, -2.0, -1.0]),
        'upper': np.array([env.quadrotor.mc * env.quadrotor.g * 1.5, 2.0, 2.0, 1.0])
    }
    
    # First test with regular MPC
    print("Testing with regular MPC first...")
    mpc = MPC(A, B, C, Q, R, horizon=10)
    env.set_controller(mpc)
    
    # Run simulation with regular MPC
    results_regular = run_simulation(env, mpc, control_constraints, max_steps=200)
    
    # Save the trajectory for comparison
    regular_trajectory = np.array(env.state_trajectory)
    
    # Reset the environment for DR-MPC test
    env = QuadrotorEnvironment(dt=0.1, enable_learning=False)
    env.obstacle_set.obstacles = []
    env.obstacle_set.add_obstacle(StaticObstacle(obstacle_position, obstacle_radius))
    env.target = np.array([5.0, 5.0, 3.0])
    env.reference_trajectory = np.array(trajectory)
    env.quadrotor.forward_dynamics = simplified_dynamics.__get__(env.quadrotor, QuadrotorModel)
    
    # Create a simple ambiguity set for the obstacle (assuming it doesn't move)
    obstacle_ambiguity_set = [
        {
            'weight': 1.0,
            'mean': np.zeros(3),
            'covariance': np.eye(3) * 0.01
        }
    ]
    
    # Now test with DR-MPC
    print("\nTesting with DR-MPC...")
    dr_mpc = DistributionallyRobustMPC(A, B, C, Q, R, horizon=10, alpha=0.9)
    env.set_controller(dr_mpc)
    
    # Run simulation with DR-MPC
    results_dr = run_simulation(env, dr_mpc, control_constraints, 
                              obstacle_position, obstacle_radius, 
                              obstacle_ambiguity_set, max_steps=200)
    
    # Save the trajectory for comparison
    dr_trajectory = np.array(env.state_trajectory)
    
    # Restore original dynamics
    env.quadrotor.forward_dynamics = original_dynamics
    
    # Visualize and compare results
    compare_results(regular_trajectory, dr_trajectory, 
                   env.target, obstacle_position, obstacle_radius)
    
    return env, results_regular, results_dr

def run_simulation(env, controller, control_constraints, 
                 obstacle_position=None, obstacle_radius=None, 
                 obstacle_ambiguity_set=None, max_steps=200):
    """Run a simulation with the given controller."""
    success = False
    collision = False
    steps = 0
    
    for step in range(max_steps):
        # Get appropriate reference trajectory
        current_idx = min(step, len(env.reference_trajectory)-1)
        end_idx = min(current_idx + controller.horizon + 1, len(env.reference_trajectory))
        reference = env.reference_trajectory[current_idx:end_idx]
        
        # Pad if needed
        if len(reference) < controller.horizon + 1:
            padding = np.array([env.target for _ in range(controller.horizon + 1 - len(reference))])
            reference = np.vstack([reference, padding])
        
        # Get control action
        if isinstance(controller, DistributionallyRobustMPC) and obstacle_position is not None:
            # Prepare obstacle info for DR-MPC
            obstacles_info = [{
                'position': obstacle_position,
                'safety_radius': obstacle_radius * 1.2,  # Add some margin
                'ambiguity_components': obstacle_ambiguity_set
            }]
            
            control = controller.solve_with_obstacles(
                env.state, reference, obstacles_info, control_constraints=control_constraints
            )
        else:
            control = controller.solve(
                env.state, reference, control_constraints=control_constraints
            )
        
        if control is None:
            print(f"Step {step}: Controller failed to find solution")
            break
        
        if step % 5 == 0:
            # Print status every 5 steps
            print(f"Step {step}:")
            print(f"  Position: {env.state[:3]}")
            print(f"  Control: {control}")
            print(f"  Distance to target: {np.linalg.norm(env.state[:3] - env.target):.4f}")
        
        # Apply control
        env.state = env.quadrotor.forward_dynamics(env.state, control)
        env.state_trajectory.append(env.state.copy())
        
        # Check for collision
        if obstacle_position is not None:
            distance = np.linalg.norm(env.state[:3] - obstacle_position)
            if distance < obstacle_radius:
                print(f"Collision detected at step {step}!")
                collision = True
                break
        
        # Check if reached target
        if np.linalg.norm(env.state[:3] - env.target) < 0.5:
            print(f"Target reached at step {step}!")
            success = True
            break
            
        steps = step + 1
    
    return {
        'success': success,
        'collision': collision,
        'steps': steps
    }

def compare_results(regular_trajectory, dr_trajectory, target, 
                   obstacle_position, obstacle_radius):
    """Compare the results of regular MPC and DR-MPC."""
    # Plot 3D trajectories
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot regular MPC trajectory
    ax.plot(regular_trajectory[:, 0], regular_trajectory[:, 1], regular_trajectory[:, 2], 
            'b-', linewidth=2, label='Regular MPC')
    
    # Plot DR-MPC trajectory
    ax.plot(dr_trajectory[:, 0], dr_trajectory[:, 1], dr_trajectory[:, 2], 
            'r-', linewidth=2, label='DR-MPC')
    
    # Plot obstacle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = obstacle_position[0] + obstacle_radius * np.cos(u) * np.sin(v)
    y = obstacle_position[1] + obstacle_radius * np.sin(u) * np.sin(v)
    z = obstacle_position[2] + obstacle_radius * np.cos(v)
    ax.plot_surface(x, y, z, color='g', alpha=0.3)
    
    # Plot start and target
    ax.scatter(regular_trajectory[0, 0], regular_trajectory[0, 1], regular_trajectory[0, 2], 
              c='blue', marker='o', s=100, label='Start')
    ax.scatter(target[0], target[1], target[2], 
              c='red', marker='*', s=200, label='Target')
    
    # Set labels and limits
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Comparison of Regular MPC and DR-MPC Trajectories')
    ax.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Plot distance to obstacle
    plt.figure(figsize=(10, 6))
    
    # Calculate distances
    regular_dists = [np.linalg.norm(pos[:3] - obstacle_position) - obstacle_radius 
                    for pos in regular_trajectory]
    dr_dists = [np.linalg.norm(pos[:3] - obstacle_position) - obstacle_radius 
               for pos in dr_trajectory]
    
    plt.plot(regular_dists, 'b-', linewidth=2, label='Regular MPC')
    plt.plot(dr_dists, 'r-', linewidth=2, label='DR-MPC')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('Step')
    plt.ylabel('Distance to Obstacle Surface [m]')
    plt.title('Distance to Obstacle During Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_with_obstacle()
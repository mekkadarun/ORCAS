import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import cvxpy as cp


# Add the parent directory to the path to import modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_models.quadrotor import QuadrotorModel
from control.mpc import MPC, DistributionallyRobustMPC
from sim.env import QuadrotorEnvironment

def test_standard_dr_mpc():
    """Test the standard distributionally robust MPC controller."""
    # Create environment
    env = QuadrotorEnvironment(dt=0.1, enable_learning=True)
    
    # Define system matrices for linear model
    n_states = 12
    n_controls = 4
    n_outputs = 3
    
    dt = 0.1
    A = np.eye(n_states)
    A[0, 3] = A[1, 4] = A[2, 5] = dt
    A[6, 9] = A[7, 10] = A[8, 11] = dt
    
    B = np.zeros((n_states, n_controls))
    B[5, 0] = dt
    B[9, 1] = dt
    B[10, 2] = dt
    B[11, 3] = dt
    
    C = np.zeros((n_outputs, n_states))
    C[0, 0] = C[1, 1] = C[2, 2] = 1
    
    Q = np.diag([10.0, 10.0, 10.0])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    
    horizon = 10
    
    # Create Distributionally Robust MPC controller
    dr_mpc = DistributionallyRobustMPC(A, B, C, Q, R, horizon=horizon, alpha=0.95)
    
    # Set controller
    env.set_controller(dr_mpc)
    
    # Run simulation
    print("\nRunning simulation with Distributionally Robust MPC...")
    results = env.run_simulation(max_steps=200)
    
    # Visualize results
    env.visualize(block=True)
    
    # Print summary
    print("\nSummary:")
    print(f"DR-MPC: {'Success' if results['success'] else 'Failed'}, " +
          f"{'Collision' if results['collision'] else 'No collision'}, " +
          f"Steps: {results['steps']}")
    
    return results

if __name__ == "__main__":
    test_standard_dr_mpc()

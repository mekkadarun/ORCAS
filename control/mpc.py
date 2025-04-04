import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional, Union


class MPC:
    """Base Model Predictive Control implementation."""
    
    def __init__(self, 
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 horizon: int = 10):
        """
        Initialize the MPC controller.
        
        Args:
            A: System matrix
            B: Control matrix
            C: Output matrix
            Q: State cost matrix
            R: Control cost matrix
            horizon: Prediction horizon
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.horizon = horizon
        
        # Get dimensions
        self.n_states = A.shape[0]
        self.n_controls = B.shape[1]
        self.n_outputs = C.shape[0]
        
    def setup_problem(self, 
                      initial_state: np.ndarray,
                      reference_trajectory: np.ndarray,
                      state_constraints: Optional[Dict] = None,
                      control_constraints: Optional[Dict] = None) -> Tuple[cp.Problem, List[cp.Variable], List[cp.Variable]]:
        """
        Set up the MPC optimization problem.
        
        Args:
            initial_state: Initial state vector
            reference_trajectory: Reference trajectory for outputs
            state_constraints: Optional state constraints
            control_constraints: Optional control constraints
            
        Returns:
            Tuple of (CVXPY problem, state variables, control variables)
        """
        # Define optimization variables
        states = [cp.Variable(self.n_states) for _ in range(self.horizon + 1)]
        controls = [cp.Variable(self.n_controls) for _ in range(self.horizon)]
        
        # Define constraints
        constraints = [states[0] == initial_state]  # Initial state constraint
        
        # State update constraints
        for t in range(self.horizon):
            constraints.append(states[t+1] == self.A @ states[t] + self.B @ controls[t])
        
        # State constraints
        if state_constraints:
            for t in range(self.horizon + 1):
                if 'lower' in state_constraints:
                    constraints.append(states[t] >= state_constraints['lower'])
                if 'upper' in state_constraints:
                    constraints.append(states[t] <= state_constraints['upper'])
        
        # Control constraints
        if control_constraints:
            for t in range(self.horizon):
                if 'lower' in control_constraints:
                    constraints.append(controls[t] >= control_constraints['lower'])
                if 'upper' in control_constraints:
                    constraints.append(controls[t] <= control_constraints['upper'])
                    
        # Define objective function
        objective = 0
        
        # Terminal state cost
        terminal_error = self.C @ states[self.horizon] - reference_trajectory[-1]
        objective += cp.quad_form(terminal_error, self.Q)
        
        # Stage costs
        for t in range(self.horizon):
            # State cost
            output_error = self.C @ states[t] - reference_trajectory[min(t, len(reference_trajectory)-1)]
            objective += cp.quad_form(output_error, self.Q)
            
            # Control cost
            objective += cp.quad_form(controls[t], self.R)
        
        # Create the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        return problem, states, controls
        
    def solve(self, 
              initial_state: np.ndarray,
              reference_trajectory: np.ndarray,
              state_constraints: Optional[Dict] = None,
              control_constraints: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Solve the MPC problem and get the optimal control input.
        
        Args:
            initial_state: Initial state vector
            reference_trajectory: Reference trajectory for outputs
            state_constraints: Optional state constraints
            control_constraints: Optional control constraints
            
        Returns:
            Optimal control input or None if infeasible
        """
        # Set up the problem
        problem, states, controls = self.setup_problem(
            initial_state, reference_trajectory, state_constraints, control_constraints
        )
        
        # Solve the problem
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Return the first control action
                return controls[0].value
            else:
                print(f"MPC optimization failed with status: {problem.status}")
                return None
                
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            return None


class DistributionallyRobustMPC(MPC):
    """
    Distributionally Robust MPC implementation using CVaR constraints.
    """
    
    def __init__(self, 
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 horizon: int = 10,
                 alpha: float = 0.95):
        """
        Initialize the DR-MPC controller.
        
        Args:
            A: System matrix
            B: Control matrix
            C: Output matrix
            Q: State cost matrix
            R: Control cost matrix
            horizon: Prediction horizon
            alpha: CVaR confidence level
        """
        super().__init__(A, B, C, Q, R, horizon)
        self.alpha = alpha
        
    def setup_problem_with_obstacles(self,
                                    initial_state: np.ndarray,
                                    reference_trajectory: np.ndarray,
                                    obstacles: List[Dict],
                                    state_constraints: Optional[Dict] = None,
                                    control_constraints: Optional[Dict] = None) -> Tuple[cp.Problem, List[cp.Variable], List[cp.Variable]]:
        """
        Set up the DR-MPC problem with obstacle avoidance.
        
        Args:
            initial_state: Initial state vector
            reference_trajectory: Reference trajectory for outputs
            obstacles: List of obstacle information dictionaries
            state_constraints: Optional state constraints
            control_constraints: Optional control constraints
            
        Returns:
            Tuple of (CVXPY problem, state variables, control variables)
        """
        from control.cvar import dist_robust_cvar_constraint
        
        # First, set up the base MPC problem
        problem, states, controls = self.setup_problem(
            initial_state, reference_trajectory, state_constraints, control_constraints
        )
        
        # Get the existing constraints and objective
        constraints = problem.constraints
        objective = problem.objective
        
        # Add CVaR constraints for obstacle avoidance
        for obstacle in obstacles:
            position = obstacle['position']
            components = obstacle['ambiguity_components']
            safety_radius_sq = obstacle['safety_radius'] ** 2
            
            for t in range(1, self.horizon + 1):  # Start from t=1 (first prediction)
                # Get the robot's position from the state
                robot_position = self.C @ states[t]  # Assuming C extracts the position
                
                # Add the distributionally robust CVaR constraint
                cvar_constraints = dist_robust_cvar_constraint(
                    robot_position, position, components, safety_radius_sq, self.alpha
                )
                
                constraints.extend(cvar_constraints)
        
        # Create the updated problem
        problem = cp.Problem(objective, constraints)
        
        return problem, states, controls
    
    def solve_with_obstacles(self,
                           initial_state: np.ndarray,
                           reference_trajectory: np.ndarray,
                           obstacles: List[Dict],
                           state_constraints: Optional[Dict] = None,
                           control_constraints: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Solve the DR-MPC problem with obstacle avoidance.
        
        Args:
            initial_state: Initial state vector
            reference_trajectory: Reference trajectory for outputs
            obstacles: List of obstacle information dictionaries
            state_constraints: Optional state constraints
            control_constraints: Optional control constraints
            
        Returns:
            Optimal control input or None if infeasible
        """
        # Set up the problem
        problem, states, controls = self.setup_problem_with_obstacles(
            initial_state, reference_trajectory, obstacles, state_constraints, control_constraints
        )
        
        # Solve the problem
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Return the first control action
                return controls[0].value
            else:
                print(f"DR-MPC optimization failed with status: {problem.status}")
                return None
                
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            return None
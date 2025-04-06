import numpy as np
import mosek
mosek.Env().putlicensepath("/home/arunmekkad/mosek/mosek.lic")
import cvxpy as cp
from typing import List, Dict, Tuple, Optional, Union
from control.cvar import dist_robust_cvar_constraint, simplified_cvar_constraint


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
            safety_radius = obstacle['safety_radius']
            
            for t in range(1, self.horizon + 1):  # Start from t=1 (first prediction)
                # Get the robot's position from the state
                robot_position = self.C @ states[t]  # Assuming C extracts the position
                
                # Add the simplified CVaR constraint
                cvar_constraints = simplified_cvar_constraint(
                    robot_position, position, components, safety_radius, self.alpha
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
           
            problem.solve(solver=cp.MOSEK, verbose=True, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8})
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Return the first control action
                return controls[0].value
            else:
                print(f"DR-MPC optimization failed with status: {problem.status}")
                return None
                
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            return None
        
    def solve_sequential_linearization(self, 
                                    initial_state: np.ndarray,
                                    reference_trajectory: np.ndarray,
                                    obstacles: List[Dict],
                                    quadrotor_model,
                                    max_iterations: int = 3,
                                    state_constraints: Optional[Dict] = None,
                                    control_constraints: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Solve using sequential linearization for nonlinear MPC.
        
        Args:
            initial_state: Initial state vector
            reference_trajectory: Reference trajectory for outputs
            obstacles: List of obstacle information
            quadrotor_model: QuadrotorModel instance for linearization
            max_iterations: Maximum number of linearization iterations
            state_constraints: Optional state constraints
            control_constraints: Optional control constraints
            
        Returns:
            Optimal control input or None if infeasible
        """
        # Initial guess for control - default to basic vertical thrust
        control = np.array([5.0, 0.0, 0.0, 0.0])
        
        # Initial state for prediction
        predicted_state = initial_state.copy()
        
        # First, try solving without sequential linearization
        try:
            print("Attempting to solve with standard method first...")
            std_control = self.solve_with_obstacles(
                initial_state, 
                reference_trajectory, 
                obstacles,
                state_constraints, 
                control_constraints
            )
            if std_control is not None:
                print("Standard solution found successfully.")
                return std_control
        except Exception as e:
            print(f"Standard solution failed with error: {e}")
        
        # If standard approach fails, try sequential linearization
        print("Attempting sequential linearization...")
        
        # Track if we ever get a valid solution
        any_success = False
        
        # Iterative linearization and solving
        for i in range(max_iterations):
            try:
                # Linearize around current prediction
                A, B = quadrotor_model.linearize(predicted_state, control)
                
                # Update system matrices
                self.A = A
                self.B = B
                
                # Solve the linearized problem
                new_control = self.solve_with_obstacles(
                    initial_state, 
                    reference_trajectory, 
                    obstacles,
                    state_constraints, 
                    control_constraints
                )
                
                if new_control is not None:
                    control = new_control
                    any_success = True
                    # Update state prediction for next linearization
                    predicted_state = quadrotor_model.forward_dynamics(predicted_state, control)
                    print(f"Iteration {i+1}: Solution found.")
                else:
                    print(f"Iteration {i+1}: No solution found.")
                    # If we already have at least one good solution, break
                    if any_success:
                        break
                    
            except Exception as e:
                print(f"Iteration {i+1} failed with error: {e}")
                # If we already have at least one good solution, break
                if any_success:
                    break
                
        return control
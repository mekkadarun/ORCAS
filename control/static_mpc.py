import cvxpy as cvx
import numpy as np

class MPC:
    def __init__(self, horizon=8, dt=0.1, quad_radius=0.3):
        self.horizon = horizon
        self.dt = dt
        self.quad_radius = quad_radius
        
        # Cost weights
        self.w_goal = 10.0
        self.w_control = 0.05
        self.w_smoothness = 0.3

    def optimize_trajectory(self, current_state, goal, obstacles):
        """Convex reformulation using linearized constraints"""
        # Control variables (accelerations)
        u = cvx.Variable((self.horizon, 2))
        
        # State variables [x, y, vx, vy]
        pos = cvx.Variable((self.horizon + 1, 2))
        vel = cvx.Variable((self.horizon + 1, 2))
        
        # Initial state constraints
        constraints = [
            pos[0] == current_state[:2],
            vel[0] == current_state[2:]
        ]
        
        # Dynamics constraints
        for t in range(self.horizon):
            constraints += [
                vel[t+1] == vel[t] + u[t] * self.dt,
                pos[t+1] == pos[t] + vel[t] * self.dt
            ]
        
        # Linearized obstacle constraints
        for t in range(1, self.horizon + 1):
            for obs in obstacles:
                obs_pos = obs.get_position()
                min_dist = self.quad_radius + obs.radius
                
                # Current relative position vector
                rel_pos = current_state[:2] - obs_pos
                norm_rel_pos = np.linalg.norm(rel_pos)
                
                if norm_rel_pos < 1e-6:  # Avoid division by zero
                    continue
                    
                # Linearization direction (unit vector)
                dir_vec = rel_pos / norm_rel_pos
                
                # Conservative linear constraint
                constraints += [
                    dir_vec @ (pos[t] - obs_pos) >= min_dist
                ]
        
        # Objective function
        goal_error = cvx.sum_squares(pos[-1] - goal)
        control_effort = cvx.sum_squares(u)
        smoothness = cvx.sum_squares(u[1:] - u[:-1])
        
        objective = cvx.Minimize(
            self.w_goal * goal_error +
            self.w_control * control_effort +
            self.w_smoothness * smoothness
        )
        
        # Solve problem
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.ECOS, verbose=False)
        
        if prob.status != cvx.OPTIMAL:
            print("Optimization failed! Using fallback.")
            return np.zeros((self.horizon, 2))
            
        return u.value

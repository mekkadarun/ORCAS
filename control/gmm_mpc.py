import cvxpy as cvx
import numpy as np
from control.cvar_constraints import CVaRObstacleAvoidance

class CVaRGMMMPC:
    def __init__(self, horizon=8, dt=0.1, quad_radius=0.3, confidence_level=0.95):
        self.horizon = horizon
        self.dt = dt
        self.quad_radius = quad_radius
        self.confidence_level = confidence_level
        self.safety_margin = 0.1
        
        self.cvar_avoidance = CVaRObstacleAvoidance(
            confidence_level=confidence_level,
            safety_margin=self.safety_margin
        )
        
        self.w_goal = 30.0
        self.w_control = 0.02
        self.w_smoothness = 0.1
        self.w_risk = 2.0

    def optimize_trajectory(self, current_state, goal, obstacles):
        u = cvx.Variable((self.horizon, 2))
        
        pos = cvx.Variable((self.horizon + 1, 2))
        vel = cvx.Variable((self.horizon + 1, 2))
        
        constraints = [
            pos[0] == current_state[:2],
            vel[0] == current_state[2:]
        ]
        
        for t in range(self.horizon):
            constraints += [
                vel[t+1] == vel[t] + u[t] * self.dt,
                pos[t+1] == pos[t] + vel[t] * self.dt + 0.5 * u[t] * self.dt**2
            ]
            
        for t in range(self.horizon + 1):
            constraints += [
                cvx.norm(vel[t]) <= 2.0
            ]
            
        for t in range(self.horizon):
            constraints += [
                cvx.norm(u[t]) <= 1.0
            ]
            
        current_dist_to_goal = np.linalg.norm(current_state[:2] - goal)
        constraints += [
            cvx.norm(pos[-1] - goal) <= 0.9 * current_dist_to_goal
        ]
                
        for t in range(1, self.horizon + 1):
            for obs in obstacles:
                min_dist = self.quad_radius + obs.radius + self.safety_margin
                
                cvar_constraints = self.cvar_avoidance.calculate_cvar_constraint(
                    pos[t], obs, t, min_dist
                )
                constraints += cvar_constraints
        
        goal_error = cvx.sum_squares(pos[-1] - goal)
        
        intermediate_goal_error = 0
        for t in range(1, self.horizon):
            weight = t / self.horizon
            intermediate_goal_error += weight * cvx.sum_squares(pos[t] - goal)
        
        control_effort = cvx.sum_squares(u)
        smoothness = cvx.sum_squares(u[1:] - u[:-1])
        
        risk_cost = 0
        for t in range(1, self.horizon + 1):
            for obs in obstacles:
                obs_pos = obs.get_position()
                dist_term = -cvx.norm(pos[t] - obs_pos)
                risk_cost += dist_term
                
        path_length = 0
        for t in range(1, self.horizon + 1):
            path_length += cvx.norm(pos[t] - pos[t-1])
        
        objective = cvx.Minimize(
            self.w_goal * goal_error +
            0.5 * self.w_goal * intermediate_goal_error +
            self.w_control * control_effort +
            self.w_smoothness * smoothness -
            self.w_risk * risk_cost +
            0.5 * path_length
        )
        
        prob = cvx.Problem(objective, constraints)
        
        try:
            prob.solve(
                solver=cvx.ECOS,
                verbose=False,
                max_iters=500,
                abstol=1e-4,
                reltol=1e-4,
                feastol=1e-4
            )
            
            if prob.status == cvx.OPTIMAL or prob.status == cvx.OPTIMAL_INACCURATE:
                return u.value
        except Exception as e:
            print(f"Optimization error: {e}")
        
        print("Using CVaR-based safe direction")
        
        to_goal = goal - current_state[:2]
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal > 0.01:
            direction = to_goal / dist_to_goal
            
            num_samples = 8
            angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
            
            min_risk = float('inf')
            safest_dir = direction
            
            for angle in angles:
                sample_dir = np.array([np.cos(angle), np.sin(angle)])
                goal_alignment = np.dot(sample_dir, direction)
                
                if goal_alignment < 0.3:
                    continue
                
                test_pos = current_state[:2] + sample_dir * self.dt * 2.0
                
                risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
                
                weighted_risk = risk * (2.0 - goal_alignment)
                
                if weighted_risk < min_risk:
                    min_risk = weighted_risk
                    safest_dir = sample_dir
            
            strength = min(1.0, dist_to_goal)
            
            control = safest_dir * strength - current_state[2:] * 0.5
            
            control_norm = np.linalg.norm(control)
            if control_norm > 1.0:
                control = control / control_norm
        else:
            control = -current_state[2:]
            
        return np.tile(control.reshape(1, 2), (self.horizon, 1))
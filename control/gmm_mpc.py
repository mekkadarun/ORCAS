import cvxpy as cvx
import numpy as np
from control.cvar_constraints import CVaRObstacleAvoidance

class CVaRGMMMPC:
    def __init__(self, horizon=10, dt=0.1, quad_radius=0.3, confidence_level=0.90):
        self.horizon = horizon
        self.dt = dt
        self.quad_radius = quad_radius
        self.confidence_level = confidence_level
        
        # Safety margin for collision avoidance
        self.safety_margin = 0.6  # Increased from 0.5
        
        self.cvar_avoidance = CVaRObstacleAvoidance(
            confidence_level=confidence_level,
            safety_margin=self.safety_margin
        )
        
        # Controller weights
        self.w_goal = 12.0        # Decreased goal weight (was 10.0)
        self.w_control = 0.8      # Weight for control effort term
        self.w_smoothness = 0.5   # Weight for control smoothness term
        self.w_risk = 18.0        # Significantly increased risk weight (was 20.0)
        
        # Flag to bypass optimization and use safe direction directly
        self.use_safe_direction = True  # Skip problematic constraints
        
        # Safety parameters
        self.safety_distance = 0.9  # Significantly increased base safety distance (was 0.7)
        self.num_direction_samples = 40  # Increased direction samples for finer coverage (was 24)
        self.min_goal_alignment = 0.1  # Allow deviation from goal direction
        
        # Risk-based velocity modulation parameters
        self.max_velocity = 1.7       # Reduced maximum velocity (was 2.0)
        self.min_velocity = 0.3       # Minimum allowed velocity
        self.risk_scaling_factor = 4.0  # Increased scaling factor (was 2.0) to slow down more in risky areas

    def optimize_trajectory(self, current_state, goal, obstacles):
        """Compute optimal control inputs for the quadrotor."""
        # If we're using safe direction approach directly, skip the optimization
        if self.use_safe_direction:
            return self.compute_safe_direction(current_state, goal, obstacles)
        
        # Try the optimization approach (this code will be skipped with use_safe_direction=True)
        try:
            u = cvx.Variable((self.horizon, 2))
            
            pos = cvx.Variable((self.horizon + 1, 2))
            vel = cvx.Variable((self.horizon + 1, 2))
            
            constraints = [
                pos[0] == current_state[:2],
                vel[0] == current_state[2:]
            ]
            
            # Dynamic constraints - motion model
            for t in range(self.horizon):
                constraints += [
                    vel[t+1] == vel[t] + u[t] * self.dt,
                    pos[t+1] == pos[t] + vel[t] * self.dt + 0.5 * u[t] * self.dt**2
                ]
            
            # Velocity constraints
            for t in range(self.horizon + 1):
                constraints += [
                    cvx.norm(vel[t]) <= 2.0
                ]
            
            # Control input constraints
            for t in range(self.horizon):
                constraints += [
                    cvx.norm(u[t]) <= 1.0
                ]
            
            # Ensure progress toward the goal
            current_dist_to_goal = np.linalg.norm(current_state[:2] - goal)
            constraints += [
                cvx.norm(pos[-1] - goal) <= 0.9 * current_dist_to_goal
            ]
            
            # Objective function components
            # Goal reaching cost
            goal_error = cvx.sum_squares(pos[-1] - goal)
            
            # Intermediate waypoints cost
            intermediate_goal_error = 0
            for t in range(1, self.horizon):
                weight = t / self.horizon
                intermediate_goal_error += weight * cvx.sum_squares(pos[t] - goal)
            
            # Control effort cost
            control_effort = cvx.sum_squares(u)
            
            # Control smoothness cost
            smoothness = cvx.sum_squares(u[1:] - u[:-1])
            
            # Risk cost (prefer paths that are far from obstacles)
            risk_cost = 0
            for t in range(1, self.horizon + 1):
                for obs in obstacles:
                    obs_pos = obs.get_position()
                    dist_term = -cvx.norm(pos[t] - obs_pos)
                    risk_cost += dist_term
            
            # Path length cost
            path_length = 0
            for t in range(1, self.horizon + 1):
                path_length += cvx.norm(pos[t] - pos[t-1])
            
            # Complete objective function
            objective = cvx.Minimize(
                self.w_goal * goal_error +
                0.5 * self.w_goal * intermediate_goal_error +
                self.w_control * control_effort +
                self.w_smoothness * smoothness -
                self.w_risk * risk_cost +
                0.5 * path_length
            )
            
            # Solve the optimization problem
            prob = cvx.Problem(objective, constraints)
            
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
        
        # Fallback to safe direction approach
        return self.compute_safe_direction(current_state, goal, obstacles)
    
    def check_collision(self, position, obstacles):
        """Check if a position is in collision with any obstacle."""
        for obs in obstacles:
            obs_pos = obs.get_position()
            
            # Get obstacle velocity magnitude to adjust safety distance
            obs_velocity = 0.0
            if hasattr(obs, 'latest_movement'):
                obs_velocity = np.linalg.norm(obs.latest_movement)
            
            # Dynamic safety distance with increased velocity factor
            velocity_factor = 1.0 + min(obs_velocity * 3.0, 2.0)  # Increased scaling
            
            # Add prediction factor - check where the obstacle will be
            prediction_time = 0.2  # Look ahead 200ms
            future_pos = obs_pos
            if hasattr(obs, 'latest_movement'):
                future_pos = obs_pos + obs.latest_movement * prediction_time
                
            # Use both current and predicted positions for collision checking
            collision_dist = self.quad_radius + obs.radius + self.safety_margin * velocity_factor
            
            # Check distance to current position
            dist_current = np.linalg.norm(position - obs_pos)
            # Check distance to predicted position
            dist_future = np.linalg.norm(position - future_pos)
            
            # Use the smaller of the two distances for safety
            dist = min(dist_current, dist_future)
            
            # Add an extra buffer for high-velocity obstacles
            if obs_velocity > 0.5:  # Only for faster obstacles
                collision_dist += obs_velocity * 0.3  # Additional safety buffer
            
            if dist <= collision_dist:
                return True
                
        return False
        
    def compute_safe_direction(self, current_state, goal, obstacles):
        """Compute a safe direction based on the CVaR risk approach."""
        print("Using CVaR-based safe direction")
        
        to_goal = goal - current_state[:2]
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal > 0.01:
            # Direction to goal
            goal_dir = to_goal / dist_to_goal
            
            # Calculate the forward half-plane (to avoid backward directions)
            # This defines the half-plane that's "forward" with respect to goal direction
            forward_normal = goal_dir  # The normal to the half-plane is the goal direction
            
            # Generate sample directions with constrained sampling to forward half-plane
            num_samples = self.num_direction_samples
            
            # Using a semicircle arc in the forward direction (+/- 90 degrees from goal)
            goal_dir_angle = np.arctan2(goal_dir[1], goal_dir[0])
            angles = []
            
            # Add samples in the forward semicircle (maximum 90 degrees deviation from goal)
            for i in range(num_samples):
                # Generate angles in range [-pi/2, pi/2] relative to goal direction
                deviation = np.pi * (i / (num_samples - 1) - 0.5)
                angles.append(goal_dir_angle + deviation)
            
            # First check if direct path to goal is collision-free
            test_pos = current_state[:2] + goal_dir * self.dt * 2.0
            goal_risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
            
            # Calculate risk at current position for velocity modulation
            current_risk = self.cvar_avoidance.calculate_total_risk(current_state[:2], obstacles)
            
            if goal_risk < 0.3 and not self.check_collision(test_pos, obstacles):
                # If direct path is safe enough, use it
                safest_dir = goal_dir
                min_risk = goal_risk
            else:
                # Otherwise, sample directions and find safest
                min_risk = float('inf')
                safest_dir = goal_dir  # Default to goal direction
                
                # Track how far we're deviating from the goal direction
                # for progress monitoring
                best_alignment = -1.0
                
                for angle in angles:
                    sample_dir = np.array([np.cos(angle), np.sin(angle)])
                    goal_alignment = np.dot(sample_dir, goal_dir)
                    
                    # Ensure we only consider directions in the forward half-plane
                    if goal_alignment < 0.0:  # Less than 0 means backwards
                        continue
                    
                    # Test position after moving in this direction
                    test_pos = current_state[:2] + sample_dir * self.dt * 2.0
                    
                    # Skip if this direction leads to collision
                    if self.check_collision(test_pos, obstacles):
                        continue
                    
                    # Calculate risk using the CVaR avoidance module
                    risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
                    
                    # Apply non-linear risk penalty for better safety
                    # Risk values close to 1.0 are severely penalized
                    # This creates a stronger barrier as risk increases
                    if risk > 0.5:
                        risk = risk * 2.0  # Double the risk for higher risk values
                    
                    # Super-penalize directions with risk above threshold
                    if risk > 0.7:
                        risk = risk * 5.0  # Extreme penalty for very high risk
                    
                    # New weighting approach prioritizing safety while maintaining progress
                    risk_weight = 4.0       # Increased from 2.0
                    alignment_weight = 0.8  # Decreased from 1.0
                    
                    # This formula prioritizes directions that are both:
                    # 1. Low risk (multiply by risk_weight to emphasize this even more)
                    # 2. Close to goal direction (higher goal_alignment is better)
                    weighted_score = (risk * risk_weight) - (goal_alignment * alignment_weight)
                    
                    if weighted_score < min_risk:
                        min_risk = weighted_score
                        safest_dir = sample_dir
                        best_alignment = goal_alignment
                
                # If all sampled directions are unsafe, try emergency directions
                # but still maintain forward progress
                if min_risk > 0.7 or best_alignment < 0.3:
                    # Generate more fine-grained samples in the forward half-plane
                    more_angles = np.linspace(goal_dir_angle - np.pi/2, goal_dir_angle + np.pi/2, num_samples*2)
                    
                    for angle in more_angles:
                        sample_dir = np.array([np.cos(angle), np.sin(angle)])
                        goal_alignment = np.dot(sample_dir, goal_dir)
                        
                        # Ensure we're still in the forward half-plane
                        if goal_alignment < 0.0:
                            continue
                            
                        # Test position
                        test_pos = current_state[:2] + sample_dir * self.dt * 2.0
                        
                        # Skip if this direction leads to collision
                        if self.check_collision(test_pos, obstacles):
                            continue
                        
                        # Calculate risk
                        risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
                        
                        # For emergency directions, we prioritize collision avoidance more,
                        # but still maintain some forward progress
                        weighted_score = (risk * 3.0) - (goal_alignment * 0.5)
                        
                        if weighted_score < min_risk:
                            min_risk = weighted_score
                            safest_dir = sample_dir
            
            # Calculate tangential force (perpendicular to obstacle direction)
            # This lets the robot "slide" along obstacles rather than being pushed directly away
            tangential_force = np.zeros(2)
            
            for obs in obstacles:
                obs_pos = obs.get_position()
                dir_to_obs = current_state[:2] - obs_pos
                dist_to_obs = np.linalg.norm(dir_to_obs)
                
                # Only consider nearby obstacles
                influence_dist = self.quad_radius + obs.radius + 1.5
                
                if dist_to_obs < influence_dist:
                    # Normalize direction
                    if dist_to_obs > 0.001:
                        dir_to_obs = dir_to_obs / dist_to_obs
                    else:
                        dir_to_obs = np.array([1.0, 0.0])
                    
                    # Calculate tangential direction (perpendicular to radial)
                    # This creates a sliding force along the obstacle's edge
                    tangent_dir = np.array([-dir_to_obs[1], dir_to_obs[0]])
                    
                    # Check which tangent direction has better goal alignment
                    if np.dot(tangent_dir, goal_dir) < 0:
                        tangent_dir = -tangent_dir
                        
                    # Scale by distance (closer = stronger)
                    strength = np.exp((influence_dist - dist_to_obs) / influence_dist) - 1.0
                    
                    # Add to total tangential force
                    tangential_force += tangent_dir * strength * 0.6
            
            # Add enhanced repulsive force for emergency collision avoidance
            repulsive_force = np.zeros(2)
            
            for obs in obstacles:
                obs_pos = obs.get_position()
                dir_to_obs = current_state[:2] - obs_pos
                dist_to_obs = np.linalg.norm(dir_to_obs)
                
                # Consider obstacle velocity for predictive avoidance
                obs_velocity = np.zeros(2)
                if hasattr(obs, 'latest_movement'):
                    obs_velocity = obs.latest_movement
                    
                # Calculate closest point of approach (CPA) for collision prediction
                relative_vel = -current_state[2:] + obs_velocity
                
                # Only consider very close obstacles for repulsion
                danger_dist = self.quad_radius + obs.radius + 1.2  # Increased from 0.8
                
                if dist_to_obs < danger_dist:
                    # Normalize direction
                    if dist_to_obs > 0.001:
                        dir_to_obs = dir_to_obs / dist_to_obs
                    else:
                        dir_to_obs = np.array([1.0, 0.0])
                    
                    # Calculate time to closest approach
                    time_to_closest = 0
                    rel_vel_mag = np.linalg.norm(relative_vel)
                    
                    if rel_vel_mag > 0.001:
                        # Project relative position onto relative velocity
                        time_to_closest = max(0, -np.dot(dir_to_obs, relative_vel) / rel_vel_mag)
                    
                    # Add more repulsion for obstacles on collision course
                    collision_factor = 1.0
                    if time_to_closest < 1.0 and time_to_closest > 0:
                        collision_factor = 2.0  # Double strength if collision likely soon
                    
                    # Calculate distance-based repulsion strength with aggressive scaling
                    # for very close obstacles
                    normalizer = danger_dist * 0.25  # Scale factor to make the repulsion stronger
                    proximity_ratio = max(0, (danger_dist - dist_to_obs)) / normalizer
                    strength = np.exp(proximity_ratio) - 1.0  # Exponential scaling
                    
                    # Combined repulsive force with collision prediction
                    repulsive_force += dir_to_obs * strength * 0.6 * collision_factor  # Increased strength
            
            # Combine forces with safest direction 
            # (with emphasis on tangential sliding for smooth avoidance)
            combined_dir = safest_dir + tangential_force + repulsive_force
            
            # Project back to forward half-plane if needed
            # This ensures we never go backwards with respect to the goal
            if np.dot(combined_dir, goal_dir) < 0:
                # Remove the backwards component
                proj = np.dot(combined_dir, goal_dir) * goal_dir
                combined_dir = combined_dir - proj
                
                # Add a small forward component
                combined_dir = combined_dir + goal_dir * 0.2
            
            # Normalize if non-zero
            combined_norm = np.linalg.norm(combined_dir)
            if combined_norm > 0.001:
                combined_dir = combined_dir / combined_norm
            else:
                combined_dir = goal_dir
            
            # Scale by distance to goal (slower near goal)
            goal_strength = min(1.0, dist_to_goal)
            
            # Enhanced velocity modulation based on risk
            # Higher risk = significantly slower with more aggressive scaling
            risk_factor = max(0.1, 1.0 - min(current_risk * 3.0, 0.9))  # More aggressive scaling
            
            # Also consider distance to nearest obstacle for speed modulation
            min_obstacle_dist = float('inf')
            for obs in obstacles:
                dist = np.linalg.norm(current_state[:2] - obs.get_position()) - (self.quad_radius + obs.radius)
                min_obstacle_dist = min(min_obstacle_dist, dist)
            
            # Create a distance-based scaling factor (smaller distance = slower speed)
            distance_scaling = min(1.0, max(0.1, min_obstacle_dist / 2.0))
            
            # Combine risk and distance factors (taking the more conservative value)
            combined_factor = min(risk_factor, distance_scaling)
            
            # Apply to speed scale
            speed_scale = self.min_velocity + (self.max_velocity - self.min_velocity) * combined_factor
            
            # Create control input (reduced dependency on current velocity for more responsive control)
            control = combined_dir * goal_strength * speed_scale - current_state[2:] * 0.3
            
            # Clip control to maximum
            control_norm = np.linalg.norm(control)
            if control_norm > 1.0:
                control = control / control_norm
        else:
            # If very close to goal, just stop
            control = -current_state[2:]
            
        return np.tile(control.reshape(1, 2), (self.horizon, 1))
import cvxpy as cvx
import numpy as np
from control.cvar_constraints import CVaRObstacleAvoidance

class RealCVaRGMMMPC_3D:
    """
    3D-optimized MPC controller using CVaR for risk assessment
    """
    def __init__(self, horizon=8, dt=0.1, quad_radius=0.3, confidence_level=0.95, safety_margin=0.6):
        self.horizon = horizon
        self.dt = dt
        self.quad_radius = quad_radius
        self.confidence_level = confidence_level
        self.safety_margin = safety_margin
        
        # Safety margin for collision avoidance
        self.safety_margin = 0.6  # Margin for 3D environment
        
        self.cvar_avoidance = CVaRObstacleAvoidance(
            confidence_level=confidence_level,
            safety_margin=self.safety_margin
        )
        
        # Controller weights
        self.w_goal = 12.0        # Weight for goal-reaching term
        self.w_control = 0.8      # Weight for control effort term
        self.w_smoothness = 0.5   # Weight for control smoothness term
        self.w_risk = 18.0        # Weight for risk term
        
        # Use sampling-based approach for simplicity and reliability
        self.use_safe_direction = True
        
        # Safety parameters
        self.safety_distance = 0.9
        self.num_direction_samples = 60  # Increased for better 3D coverage
        self.min_goal_alignment = 0.1
        
        # Velocity modulation parameters
        self.max_velocity = 1.7      
        self.min_velocity = 0.3       
        self.risk_scaling_factor = 4.0
        
        # 3D specific parameters
        self.altitude_gain = 2.5      # Higher priority for altitude control (increased)
        self.altitude_threshold = 0.2  # Threshold for altitude error (decreased)
        
    def optimize_trajectory(self, current_state, goal, obstacles):
        """
        Compute optimal control inputs for the quadrotor in 3D.
        
        Args:
            current_state: Current state of the quadrotor [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
            goal: Goal position [x, y, z]
            obstacles: List of obstacle objects
            
        Returns:
            Optimal control inputs for the horizon
        """
        for obs in obstacles:
            if hasattr(obs, 'set_confidence_level'):
                obs.set_confidence_level(self.confidence_level)
        # Using safe direction approach for simplicity and reliability
        return self.compute_safe_direction_3d(current_state, goal, obstacles)
        
    def check_collision(self, position, obstacles):
        """
        Check if a position is in collision with any obstacle.
        Properly handles 3D positions and obstacles.
        
        Args:
            position: Position to check [x, y, z]
            obstacles: List of obstacles
            
        Returns:
            True if in collision, False otherwise
        """
        for obs in obstacles:
            obs_pos = obs.get_position()
            
            # Ensure we're comparing in the same dimension
            if len(position) == 3 and len(obs_pos) == 2:
                # If position is 3D but obstacle is 2D, extend obstacle to 3D
                obs_pos_3d = np.append(obs_pos, position[2])
                dist = np.linalg.norm(position - obs_pos_3d)
            elif len(position) == 2 and len(obs_pos) == 3:
                # If position is 2D but obstacle is 3D, project obstacle to 2D
                dist = np.linalg.norm(position - obs_pos[:2])
            else:
                # Same dimensions, direct comparison
                dist = np.linalg.norm(position - obs_pos)
            
            # Get obstacle velocity magnitude to adjust safety distance
            obs_velocity = 0.0
            if hasattr(obs, 'latest_movement'):
                obs_velocity = np.linalg.norm(obs.latest_movement)
            
            # Dynamic safety distance with velocity factor
            velocity_factor = 1.0 + min(obs_velocity * 2.0, 1.5)
            
            # Compute total collision distance
            collision_dist = self.quad_radius + obs.radius + self.safety_margin * velocity_factor
            
            # Add extra buffer for high-velocity obstacles
            if obs_velocity > 0.5:
                collision_dist += obs_velocity * 0.2
            
            if dist <= collision_dist:
                return True
                
        return False
        
    def compute_safe_direction_3d(self, current_state, goal, obstacles):
        """
        Compute a safe direction based on the CVaR risk approach,
        optimized for true 3D environment.
        
        Args:
            current_state: Current state [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
            goal: Goal position [x, y, z]
            obstacles: List of obstacles
            
        Returns:
            Control inputs [thrust, roll_torque, pitch_torque, yaw_torque]
        """
        # Extract position and velocity
        pos = current_state[:3]
        vel = current_state[3:6]
        angles = current_state[6:9]
        ang_vel = current_state[9:12]
        
        # Vector to goal and distance - use full 3D goal
        to_goal = goal - pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal > 0.01:
            # Direction to goal
            goal_dir = to_goal / dist_to_goal
            
            # Calculate risk at current position
            current_risk = self.cvar_avoidance.calculate_total_risk(pos, obstacles)
            
            # First check if direct path to goal is collision-free
            test_pos = pos + goal_dir * self.dt * 2.0
            
            goal_risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
            
            # Check altitude error specifically
            altitude_error = goal[2] - pos[2]
            has_altitude_error = abs(altitude_error) > self.altitude_threshold
            
            if goal_risk < 0.3 and not self.check_collision(test_pos, obstacles):
                # Direct path is safe, use it (but may still need altitude correction)
                if has_altitude_error:
                    # Add strong vertical component if altitude error is significant
                    vertical_dir = np.array([0, 0, np.sign(altitude_error)])
                    # Blend horizontal and vertical directions
                    horiz_weight = 0.6  # Still maintain reasonable horizontal progress
                    altitude_weight = 0.4 * self.altitude_gain  # Apply altitude gain
                    desired_dir = horiz_weight * goal_dir + altitude_weight * vertical_dir
                    # Normalize
                    desired_dir = desired_dir / np.linalg.norm(desired_dir)
                else:
                    # No altitude correction needed
                    desired_dir = goal_dir
                min_risk = goal_risk
            else:
                # Sample directions to find safest path
                min_risk = float('inf')
                desired_dir = goal_dir  # Default
                best_alignment = -1.0
                
                # Generate sample directions in 3D with increased priority on altitude
                num_samples = self.num_direction_samples
                
                # Create a basis with z-axis aligned with goal direction
                z_basis = goal_dir
                
                # Create x and y basis vectors orthogonal to z_basis
                world_up = np.array([0, 0, 1])
                if np.abs(np.dot(z_basis, world_up)) > 0.99:
                    # If goal is nearly vertical, use a different reference
                    world_up = np.array([0, 1, 0])
                
                x_basis = np.cross(z_basis, world_up)
                if np.linalg.norm(x_basis) < 1e-6:
                    # Fallback if cross product is near zero
                    x_basis = np.array([1, 0, 0])
                x_basis = x_basis / np.linalg.norm(x_basis)
                y_basis = np.cross(z_basis, x_basis)
                
                # Sample directions on the hemisphere with greater density in vertical direction
                sample_dirs = []
                
                # Add the goal direction
                sample_dirs.append(goal_dir)
                
                # Add a straight vertical direction if altitude error is significant
                if has_altitude_error:
                    vertical_dir = np.array([0, 0, np.sign(altitude_error)])
                    sample_dirs.append(vertical_dir)
                    
                    # Add some diagonals that prioritize vertical movement
                    for i in range(4):
                        angle = i * np.pi/2  # 0, 90, 180, 270 degrees
                        horiz_component = 0.4  # Reduced horizontal component
                        vert_component = 0.6 * np.sign(altitude_error)  # Increased vertical component
                        diag = np.array([
                            horiz_component * np.cos(angle),
                            horiz_component * np.sin(angle),
                            vert_component
                        ])
                        diag = diag / np.linalg.norm(diag)
                        sample_dirs.append(diag)
                
                # Add samples on the hemisphere
                for i in range(num_samples - len(sample_dirs)):
                    # Modified sampling to prioritize z-direction for altitude control
                    if has_altitude_error and np.random.random() < 0.4:
                        # 40% chance to focus on vertical movement when needed
                        theta = (np.pi/6) * np.random.random()  # Very small angle from vertical
                        if altitude_error < 0:
                            theta = np.pi - theta  # Point downward if need to descend
                    else:
                        # Standard sampling in hemisphere
                        theta = (np.pi/2) * np.sqrt(np.random.random())  # More samples near goal direction
                    
                    phi = 2 * np.pi * np.random.random()
                    
                    # Convert to Cartesian coordinates in the goal-oriented basis
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                    
                    # Transform to world coordinates
                    dir_sample = x * x_basis + y * y_basis + z * z_basis
                    dir_sample = dir_sample / np.linalg.norm(dir_sample)
                    
                    sample_dirs.append(dir_sample)
                
                # Evaluate each direction
                for sample_dir in sample_dirs:
                    # Calculate alignment with goal
                    goal_alignment = np.dot(sample_dir, goal_dir)
                    
                    # Calculate alignment with vertical direction if altitude error exists
                    vertical_alignment = 0
                    if has_altitude_error:
                        vertical_dir = np.array([0, 0, np.sign(altitude_error)])
                        vertical_alignment = np.dot(sample_dir, vertical_dir)
                    
                    # Test position
                    test_pos = pos + sample_dir * self.dt * 2.0
                    
                    # Skip if collision
                    if self.check_collision(test_pos, obstacles):
                        continue
                    
                    # Calculate risk
                    risk = self.cvar_avoidance.calculate_total_risk(test_pos, obstacles)
                    
                    # Apply non-linear risk penalty for better safety
                    if risk > 0.5:
                        risk = risk * 2.0
                    
                    # Super-penalize directions with risk above threshold
                    if risk > 0.7:
                        risk = risk * 5.0
                    
                    # Combined score considering:
                    # 1. Risk avoidance
                    # 2. Goal alignment
                    # 3. Vertical alignment (when needed)
                    risk_weight = 4.0
                    alignment_weight = 0.8
                    vertical_weight = 0.0
                    
                    # Apply vertical weight only when altitude error is significant
                    if has_altitude_error:
                        vertical_weight = self.altitude_gain  # Prioritize vertical alignment
                    
                    weighted_score = (risk * risk_weight) - (goal_alignment * alignment_weight) - (vertical_alignment * vertical_weight)
                    
                    if weighted_score < min_risk:
                        min_risk = weighted_score
                        desired_dir = sample_dir
                        best_alignment = goal_alignment
            
            # Calculate velocity modulation based on risk and distance to obstacles
            risk_factor = max(0.1, 1.0 - min(current_risk * 3.0, 0.9))
            
            # Consider distance to nearest obstacle
            min_obstacle_dist = float('inf')
            for obs in obstacles:
                obs_pos = obs.get_position()
                
                # Handle dimensionality differences
                if len(obs_pos) != 3:
                    # Extend 2D obstacle position to 3D by using quadrotor's z coordinate
                    obs_pos_3d = np.append(obs_pos, pos[2])
                else:
                    obs_pos_3d = obs_pos
                
                dist = np.linalg.norm(pos - obs_pos_3d) - (self.quad_radius + obs.radius)
                min_obstacle_dist = min(min_obstacle_dist, dist)
            
            # Distance-based scaling
            distance_scaling = min(1.0, max(0.1, min_obstacle_dist / 2.0))
            
            # Combined scaling factor
            combined_factor = min(risk_factor, distance_scaling)
            
            # Calculate desired velocity with a more conservative speed
            speed_scale = self.min_velocity + (self.max_velocity - self.min_velocity) * combined_factor
            
            # Special case for altitude correction
            if has_altitude_error and abs(altitude_error) > 0.5:
                # If we're far from the desired altitude, prioritize vertical movement
                desired_dir = np.array([
                    desired_dir[0] * 0.4,  # Reduce horizontal components
                    desired_dir[1] * 0.4,
                    desired_dir[2] * 1.5   # Boost vertical component
                ])
                # Normalize the direction
                norm = np.linalg.norm(desired_dir)
                if norm > 0:
                    desired_dir = desired_dir / norm
                
                # Increase vertical speed if needed
                if abs(altitude_error) > 1.0:
                    speed_scale = min(self.max_velocity, speed_scale * 1.2)  # Boost speed for altitude correction
            
            # Calculate desired velocity with direction
            desired_vel = desired_dir * speed_scale
            
            # Add strong damping on velocity to prevent oscillations
            velocity_error = desired_vel - vel
            
            # Physical parameters
            m = 1.0  # Mass
            g = 9.81  # Gravity
            
            # Current angles
            phi = angles[0]      # Roll
            theta = angles[1]    # Pitch
            psi = angles[2]      # Yaw
            
            # Current angular rates
            phi_dot = ang_vel[0]
            theta_dot = ang_vel[1]
            psi_dot = ang_vel[2]
            
            # Desired acceleration with damping
            accel_damping = 0.7  # Damping factor
            desired_accel = velocity_error / (self.dt * 2.0) * accel_damping
            
            # For altitude control, use a dedicated PID controller
            # This improves stability significantly
            Kp_z = 2.5  # Increased proportional gain for altitude
            Kd_z = 1.2  # Increased derivative gain for altitude
            
            # Fix z-axis inversion: In our system, positive z is upward
            # but in the quadrotor model, downward force is needed to increase altitude
            z_error = goal[2] - pos[2]  # Positive when need to go up
            z_vel_error = 0 - vel[2]    # Target zero vertical velocity at goal
            
            # The negative sign here is crucial - it inverts the acceleration 
            # direction to match the quadrotor physics model
            desired_z_accel = -(Kp_z * z_error + Kd_z * z_vel_error)
            
            # Calculate thrust to achieve desired z-acceleration
            cos_phi = np.cos(phi)
            cos_theta = np.cos(theta)
            
            # Calculate thrust needed to counteract gravity and achieve desired z-acceleration
            thrust = m * (g - desired_z_accel) / (cos_phi * cos_theta)
            
            # Ensure thrust is positive and within limits
            thrust = np.clip(thrust, 0.1 * m * g, 2.0 * m * g)
            
            # PID controller for attitude control
            # Calculate desired roll and pitch angles to achieve desired x-y acceleration
            desired_theta = desired_accel[0] / g
            desired_phi = -desired_accel[1] / g
            
            # Limit desired angles to prevent instability
            max_angle = 0.3  # About 17 degrees
            desired_phi = np.clip(desired_phi, -max_angle, max_angle)
            desired_theta = np.clip(desired_theta, -max_angle, max_angle)
            
            # PD controller for attitude
            Kp_attitude = 3.0  # Proportional gain
            Kd_attitude = 1.5  # Derivative gain
            
            # Compute torques with PD control
            roll_torque = Kp_attitude * (desired_phi - phi) - Kd_attitude * phi_dot
            pitch_torque = Kp_attitude * (desired_theta - theta) - Kd_attitude * theta_dot
            
            # For yaw, just try to maintain zero yaw
            yaw_torque = -Kp_attitude * psi - Kd_attitude * psi_dot
            
            # Clip torques
            max_torque = 0.5
            roll_torque = np.clip(roll_torque, -max_torque, max_torque)
            pitch_torque = np.clip(pitch_torque, -max_torque, max_torque)
            yaw_torque = np.clip(yaw_torque, -max_torque, max_torque)
            
            # Return control inputs [thrust, roll_torque, pitch_torque, yaw_torque]
            control = np.array([thrust, roll_torque, pitch_torque, yaw_torque])
            
            # Return the same control for all time steps in the horizon
            return np.tile(control.reshape(1, 4), (self.horizon, 1))
            
        else:
            # Very close to goal, hover
            # Improved hover control - stabilize with stronger position hold
            
            # Physical parameters
            m = 1.0  # Mass
            g = 9.81  # Gravity
            
            # Current angles
            phi = current_state[6]
            theta = current_state[7]
            psi = current_state[8]
            
            # Current angular velocities
            phi_dot = current_state[9]
            theta_dot = current_state[10]
            psi_dot = current_state[11]
            
            # PD controller for position hold
            Kp_pos = 2.0
            Kd_vel = 1.5
            
            # Position error
            pos_error = goal - pos
            
            # Desired accelerations
            desired_accel_x = Kp_pos * pos_error[0] - Kd_vel * vel[0]
            desired_accel_y = Kp_pos * pos_error[1] - Kd_vel * vel[1]
            desired_accel_z = Kp_pos * pos_error[2] - Kd_vel * vel[2]
            
            # Calculate desired angles
            desired_theta = desired_accel_x / g
            desired_phi = -desired_accel_y / g
            
            # Limit angles
            max_angle = 0.2  # Lower for hover
            desired_phi = np.clip(desired_phi, -max_angle, max_angle)
            desired_theta = np.clip(desired_theta, -max_angle, max_angle)
            
            # Calculate thrust (accounting for orientation)
            cos_phi = np.cos(phi)
            cos_theta = np.cos(theta)
            thrust = m * (g - desired_accel_z) / (cos_phi * cos_theta)
            thrust = np.clip(thrust, 0.8 * m * g, 1.2 * m * g)
            
            # Attitude PD controller
            Kp_attitude = 3.0
            Kd_attitude = 2.0  # More damping in hover
            
            roll_torque = Kp_attitude * (desired_phi - phi) - Kd_attitude * phi_dot
            pitch_torque = Kp_attitude * (desired_theta - theta) - Kd_attitude * theta_dot
            yaw_torque = -Kp_attitude * psi - Kd_attitude * psi_dot
            
            # Clip torques
            max_torque = 0.4  # Reduced for hover
            roll_torque = np.clip(roll_torque, -max_torque, max_torque)
            pitch_torque = np.clip(pitch_torque, -max_torque, max_torque)
            yaw_torque = np.clip(yaw_torque, -max_torque, max_torque)
            
            control = np.array([thrust, roll_torque, pitch_torque, yaw_torque])
            return np.tile(control.reshape(1, 4), (self.horizon, 1))
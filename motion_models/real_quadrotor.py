import numpy as np

class RealQuadrotor:
    def __init__(self, initial_state=None):
        """Initialize a realistic 3D quadrotor model as described in the paper"""
        # State: [x, y, z, ẋ, ẏ, ż, φ, θ, ψ, φ̇, θ̇, ψ̇]
        # Where:
        # x, y, z: position
        # ẋ, ẏ, ż: velocity
        # φ, θ, ψ: roll, pitch, yaw angles
        # φ̇, θ̇, ψ̇: angular velocities
        
        if initial_state is None:
            self.state = np.zeros(12)
        else:
            self.state = np.array(initial_state, dtype=float)
            
        # Physical parameters (can be adjusted)
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.m = 1.0   # Mass (kg)
        self.l = 0.2   # Distance from CoM to rotors (m)
        self.Ixx = 0.01  # Moment of inertia around x-axis (kg·m²)
        self.Iyy = 0.01  # Moment of inertia around y-axis (kg·m²)
        self.Izz = 0.02  # Moment of inertia around z-axis (kg·m²)
        
        # Control input limits
        self.max_thrust = 2.0 * self.m * self.g  # Maximum thrust (N)
        self.max_torque = 0.5  # Maximum torque (N·m)
        
        # Physical size for collision detection
        self.radius = 0.3  # Radius for collision detection (m)
        
        # Trajectory history
        self.trajectory = []
        
    def update_state(self, control_input, dt):
        """Update the quadrotor state based on control inputs and dynamics"""
        # Clip control inputs to respect physical limits
        u1 = np.clip(control_input[0], 0, self.max_thrust)
        u2 = np.clip(control_input[1], -self.max_torque, self.max_torque)
        u3 = np.clip(control_input[2], -self.max_torque, self.max_torque)
        u4 = np.clip(control_input[3], -self.max_torque, self.max_torque)
        
        # Extract current state
        x, y, z = self.state[0:3]
        x_dot, y_dot, z_dot = self.state[3:6]
        phi, theta, psi = self.state[6:9]
        phi_dot, theta_dot, psi_dot = self.state[9:12]
        
        # Compute accelerations according to the quadrotor dynamics
        # Translational dynamics (as per the paper)
        x_ddot = self.g * theta
        y_ddot = -self.g * phi
        # The z-axis force is now correctly oriented
        # Positive thrust (u1) produces upward acceleration (negative z_ddot)
        z_ddot = u1 / self.m - self.g  # Added gravity and corrected sign
        
        # Rotational dynamics (as per the paper)
        phi_ddot = self.l / self.Ixx * u2
        theta_ddot = self.l / self.Iyy * u3
        psi_ddot = self.l / self.Izz * u4
        
        # Update velocities
        x_dot_new = x_dot + x_ddot * dt
        y_dot_new = y_dot + y_ddot * dt
        z_dot_new = z_dot + z_ddot * dt
        phi_dot_new = phi_dot + phi_ddot * dt
        theta_dot_new = theta_dot + theta_ddot * dt
        psi_dot_new = psi_dot + psi_ddot * dt
        
        # Update positions and angles
        x_new = x + x_dot * dt + 0.5 * x_ddot * dt**2
        y_new = y + y_dot * dt + 0.5 * y_ddot * dt**2
        z_new = z + z_dot * dt + 0.5 * z_ddot * dt**2
        phi_new = phi + phi_dot * dt + 0.5 * phi_ddot * dt**2
        theta_new = theta + theta_dot * dt + 0.5 * theta_ddot * dt**2
        psi_new = psi + psi_dot * dt + 0.5 * psi_ddot * dt**2
        
        # Update the state
        self.state = np.array([
            x_new, y_new, z_new,
            x_dot_new, y_dot_new, z_dot_new,
            phi_new, theta_new, psi_new,
            phi_dot_new, theta_dot_new, psi_dot_new
        ])
        
        # Record trajectory
        self.trajectory.append(self.state[:3].copy())
        
    def get_state(self):
        """Return the full state vector"""
        return self.state.copy()
    
    def get_position(self):
        """Return the position coordinates [x, y, z]"""
        return self.state[:3].copy()
    
    def get_orientation(self):
        """Return the orientation angles [φ, θ, ψ]"""
        return self.state[6:9].copy()
    
    def get_velocity(self):
        """Return the velocity vector [ẋ, ẏ, ż]"""
        return self.state[3:6].copy()
    
    def get_angular_velocity(self):
        """Return the angular velocity vector [φ̇, θ̇, ψ̇]"""
        return self.state[9:12].copy()
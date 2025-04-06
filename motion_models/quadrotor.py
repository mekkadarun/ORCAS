import numpy as np

class QuadrotorModel:
    # Implements the quadrotor motion model presented in the paper

    def __init__(self, dt, g=9.81, mc=1.0, lo=0.2, Ixx=1.0, Iyy=1.0, Izz=1.0):
        """
        dt  - timestep (s)
        g   - Acceleration due to gravity (m/s^2)
        mc  - Mass (kg)
        lo  - Distance between center of mass and quadrotor (m)
        Ixx - Moment of inertia along x-axis (kgm^2)
        Iyy - Moment of inertia along y-axis (kgm^2)
        Izz - Moment of inertia along z-axis (kgm^2)
        """
        self.dt = dt
        self.g = g
        self.mc = mc
        self.lo = lo
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

    def forward_dynamics(self, state, control):
        
        x,y,z, x_dot,y_dot,z_dot, phi,theta,psi, phi_dot,theta_dot,psi_dot = state
        u1,u2,u3,u4 = control

        
        x_ddot = self.g * theta
        y_ddot = -self.g * phi
        z_ddot = (1/self.mc) * u1
        
        phi_ddot = (self.lo/self.Ixx) * u2
        theta_ddot = (self.lo/self.Iyy) * u3 
        psi_ddot = (self.lo/self.Izz) * u4


        # Discrete-time integration (Euler method)
        x_dot_next = x_dot + x_ddot * self.dt
        y_dot_next = y_dot + y_ddot * self.dt
        z_dot_next = z_dot + z_ddot * self.dt

        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        z_next = z + z_dot * self.dt

        phi_dot_next = phi_dot + phi_ddot * self.dt
        theta_dot_next = theta_dot + theta_ddot * self.dt
        psi_dot_next = psi_dot + psi_ddot * self.dt

        phi_next = phi + phi_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        psi_next = psi + psi_dot * self.dt

        next_state = np.array([
            x_next, y_next, z_next, x_dot_next, y_dot_next, z_dot_next,
            phi_next, theta_next, psi_next, phi_dot_next, theta_dot_next, psi_dot_next
        ])

        return next_state
    

    def linearize(self, state, control):
        """
        Linearize the quadrotor dynamics around an operating point.
        
        Args:
            state: Current state vector
            control: Current control input
            
        Returns:
            A: State matrix
            B: Control matrix
        """
        # If control is None, use a default control
        if control is None:
            control = np.zeros(4)
            
        # Numerical differentiation for Jacobians
        eps = 1e-6
        A = np.zeros((12, 12))
        B = np.zeros((12, 4))
        
        # Compute baseline next state
        next_state = self.forward_dynamics(state, control)
        
        # Compute A matrix (state derivatives)
        for i in range(12):
            perturbed_state = state.copy()
            perturbed_state[i] += eps
            perturbed_next = self.forward_dynamics(perturbed_state, control)
            A[:, i] = (perturbed_next - next_state) / eps
        
        # Compute B matrix (control derivatives)
        for i in range(4):
            perturbed_control = control.copy()
            perturbed_control[i] += eps
            perturbed_next = self.forward_dynamics(state, perturbed_control)
            B[:, i] = (perturbed_next - next_state) / eps
        
        return A, B
    
    def simplified_dynamics(self, state, control):
        """
        A simplified version of the dynamics that's easier to control.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
            control: Control input [thrust, roll_rate, pitch_rate, yaw_rate]
            
        Returns:
            Next state
        """
        x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot = state
        u1, u2, u3, u4 = control
        
        # Direct control of velocities (simplified model)
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
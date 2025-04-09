import numpy as np

class Quadrotor:
    def __init__(self, initial_state):
        self.state = initial_state.astype(float)  # [x, y, vx, vy]
        self.radius = 0.3
        self.max_accel = 2.0
        self.trajectory = []  # For tracking history
        
    def update_state(self, acceleration, dt):
        # Clip acceleration first
        acceleration = np.clip(acceleration, 
                             -self.max_accel, 
                             self.max_accel)
        
        # Store original velocity for position calculation
        original_vel = self.state[2:].copy()
        
        # Update velocity FIRST
        self.state[2:] += acceleration * dt
        
        # Update position using ORIGINAL velocity + acceleration effect
        self.state[:2] += original_vel * dt + 0.5 * acceleration * dt**2
        
        # Record trajectory
        self.trajectory.append(self.state[:2].copy())
        
    def get_state(self):
        return self.state.copy()
    
    def get_position(self):
        return self.state[:2]

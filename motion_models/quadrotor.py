import numpy as np

class Quadrotor:
    def __init__(self, initial_state):
        self.radius = 0.3 
        # State: [x, y, vx, vy]
        self.state = initial_state.astype(float)
        self.radius = 0.3  # Collision radius of quadrotor
        self.max_accel = 2.0  # m/s²
        
    def update_state(self, acceleration, dt):
        # Use proper kinematic equations
        self.state[2:] += acceleration * dt
        self.state[:2] += self.state[2:] * dt + 0.5 * acceleration * dt**2

        
        # Update velocity and position
        self.state[2:] += acceleration * dt
        self.state[:2] += self.state[2:] * dt
        
    def get_state(self):
        return self.state.copy()
        
    def get_position(self):
        return self.state[:2]

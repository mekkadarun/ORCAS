import numpy as np

class DynamicObstacle:
    def __init__(self, position, velocity, radius=0.5):
        """Initialize a dynamic obstacle with position, velocity and radius"""
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        
    def update_position(self, dt):
        """Update actual position with floating-point precision"""
        self.position += self.velocity * dt
        
    def get_predicted_positions(self, dt, horizon):
        """Return predicted future positions as float array"""
        return [self.position + self.velocity * t * dt 
                for t in range(1, horizon+1)]
    
    def get_position(self):
        """Return current position"""
        return self.position.copy()
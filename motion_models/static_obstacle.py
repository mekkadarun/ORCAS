import numpy as np

class Obstacle:
    def __init__(self, position, radius=0.5):
        self.position = np.array(position)
        self.radius = radius
        
    def get_position(self):
        return self.position.copy()
        
    def get_constraint_parameters(self):
        """Return parameters needed for collision avoidance constraint"""
        return {
            'center': self.position,
            'radius': self.radius
        }
        
    def distance_to(self, point):
        """Calculate distance from point to obstacle surface"""
        return np.linalg.norm(point - self.position) - self.radius

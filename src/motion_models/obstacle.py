import numpy as np
from typing import List, Tuple, Optional

class Obstacle:
    
    def __init__(self, position: np,ndarray, radius: float):

        # Initialize an obstalce

        # position : Initial position of obstacle's CoM
        # radius : Radius of the obstacle

        self.position = np.array(position, dtype=float)
        self.radius = float(radius)
        self.trajectory = [self.position.copy()] # storing movement history

    def get_position(self) -> np.ndarray:
        return self.position
    
    def get_radius(self) -> float:
        return self.radius
    
    def get_trajectory(self) -> List[np.ndarray]:
        return self.trajectory
    
    def distance_to(self, point: np.ndarray) -> float:

        # Returns the distance to point from the obstacle

        center_distance = np.linalg.norm(point - self.position)
        return max(0.0, center_distance - self.radius)
    
    
    
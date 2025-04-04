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
    
    def is_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:

        # Check if the point is in collision with the obstacle
        return self.distance_to(point) <= safety_margin
    
    def update(self, dt: float):
        raise NotImplementedError("Subclasses must implement update()")

class StaticObstacle(Obstacle):

    def update(self, dt: float):
        pass                        # since no movement

class DynamicObstacle(Obstacle):

    def __init__(self, position: np.ndarray, radius):
        super().__init__(position, radius)
        self.movements = [] # Store previous movement vectors

    def get_movements(self) -> List[np.ndarray]:
        return self.movements
    
    def move(self, translation: np.ndarray):

        # Move the obstacle by given translation vector
        self.movements.append(translation.copy())
        self.position += translation
        self.trajectory.append(self.position.copy())

class GaussianObstacle(DynamicObstacle):
    
    # Obstacle that moves according to a Gaussian Distribution

    def __init__(self, position, radius, mean: np.ndarray, covariance: np.ndarray):
        super().__init__(position, radius)

        self.mean = np.array(mean, dtype=float)
        self.covariance = np.array(covariance, dtype=float)

    def update(self, dt: float):
        scaled_mean = self.mean * dt
        scaled_covariance = self.covariance * dt

        movement = np.random.multivariate_normal(scaled_mean,scaled_covariance) # Sample a random movement
        self.move(movement) # Apply the movement

class GMMObstacle(DynamicObstacle):
    pass

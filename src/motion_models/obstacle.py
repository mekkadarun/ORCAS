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

class GMMMixingObstacle(DynamicObstacle):
    
    # Simulate multimodal movement patterns
    def __init__(self, position: np.ndarray, radius:float, 
                means: List[np.ndarray], covariances: List[np.ndarray],
                weights: List[float]):
        super().__init__(position, radius)

        weights = np.array(weights, dtype=float)
        weights = weights/ np.sum(weights)

        self.n_components = len(weights)
        self.means = [np.array(mean, dtype=float) for mean in means]
        self.covariances= [np.array(cov, dtype=float) for cov in covariances]
        self.weights = weights

    def update(self, dt: float):

        component_idx = np.random.choice(self.n_components, p = self.weights)

        mean = self.means[component_idx] * dt
        cov = self.covariances[component_idx] * dt

        movement = np.random.multivariate_normal(mean, cov)

        self.move(movement)

class TimeVaryingGaussianObstacle(DynamicObstacle):

    def __init__(self, position: np.ndarray, radius: float,
                 initial_mean: np.ndarray, initial_cov: np.ndarray,
                 variance_growth_rate: float = 0.0):
        
        super().__init__(position, radius)
        self.mean = np.array(initial_mean, dtype=float)
        self.covariance = np.array(initial_cov, dtype=float)
        self.variance_growth_rate = variance_growth_rate
        self.time = 0.0

    def update(self, dt: float):

        self.time += dt

        current_cov = self.covariance + self.variance_growth_rate * self.time

        scaled_mean = self.mean * dt
        scaled_cov = current_cov * dt
        movement = np.random.multivariate_normal(scaled_mean, scaled_cov)

        self.move(movement)

class ObstacleSet:

    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        self.obstacles = obstacles if obstacles is not None else []

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles
    
    def update_all(self, dt: float):
        for obstacle in self.obstacles:
            obstacle.update(dt)

    def check_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        return any(obs.is_collision(point, safety_margin) for obs in self.obstacles)
    
    def collect_movements(self) -> List[np.ndarray]:
        movements = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, DynamicObstacle):
                movements.extend(obstacle.get_movements())
        return movements
    
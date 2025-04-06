# models/obstacle.py

import numpy as np
from typing import List, Tuple, Optional


class Obstacle:
    """Base class for all obstacles in the environment."""
    
    def __init__(self, position, radius):
        """
        Initialize an obstacle.
        
        Args:
            position: Initial position of the obstacle's center of mass (3D vector)
            radius: Radius of the obstacle (assumes spherical obstacles)
        """
        self.position = np.array(position, dtype=float)
        self.radius = float(radius)
        self.trajectory = [self.position.copy()]

    def get_position(self) -> np.ndarray:
        """Return the current position of the obstacle."""
        return self.position
    
    def get_radius(self) -> float:
        """Return the radius of the obstacle."""
        return self.radius
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Return the trajectory history of the obstacle."""
        return self.trajectory
    
    def distance_to(self, point: np.ndarray) -> float:
        """
        Calculate the Euclidean distance from the obstacle surface to a point.
        
        Args:
            point: 3D position to calculate distance to
            
        Returns:
            Distance from the obstacle surface to the point
        """
        center_distance = np.linalg.norm(point - self.position)
        return max(0.0, center_distance - self.radius)
    
    def is_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """
        Check if a point is in collision with the obstacle (including safety margin).
        
        Args:
            point: 3D position to check for collision
            safety_margin: Additional safety distance beyond physical radius
            
        Returns:
            True if collision detected, False otherwise
        """
        return self.distance_to(point) <= safety_margin
    
    def update(self, dt: float):
        """
        Update the obstacle state over time step dt.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    
class StaticObstacle(Obstacle):
    """A static obstacle that doesn't move."""
    
    def update(self, dt: float):
        """No movement for static obstacles."""
        pass


class DynamicObstacle(Obstacle):
    """Base class for obstacles that move in the environment."""
    
    def __init__(self, position, radius):
        """
        Initialize a dynamic obstacle.
        
        Args:
            position: Initial position (3D vector)
            radius: Obstacle radius
        """
        super().__init__(position, radius)
        self.movements = []
        
                
    def get_movements(self) -> List[np.ndarray]:
        """Return the history of movement vectors."""
        return self.movements
    
    def move(self, translation: np.ndarray):
        """
        Move the obstacle by the given translation vector.
        
        Args:
            translation: Vector to translate the obstacle by
        """
        self.movements.append(translation.copy())
        self.position = self.position + translation
        self.trajectory.append(self.position.copy())


class GaussianObstacle(DynamicObstacle):
    """
    An obstacle that moves according to a Gaussian distribution.
    Used to simulate simple random movements.
    """
    
    def __init__(self, position: np.ndarray, radius: float, 
                 mean: np.ndarray, covariance: np.ndarray):
        """
        Initialize a Gaussian-moving obstacle.
        
        Args:
            position: Initial position (3D vector)
            radius: Obstacle radius
            mean: Mean of the Gaussian movement distribution
            covariance: Covariance matrix of the Gaussian distribution
        """
        super().__init__(position, radius)
        self.mean = np.array(mean, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        
    def update(self, dt: float):
        """
        Update the obstacle position using a random sample from Gaussian.
        
        Args:
            dt: Time step
        """
        # Scale the mean and covariance by dt
        scaled_mean = self.mean * dt
        scaled_cov = self.covariance * dt
        
        # Sample a random movement
        movement = np.random.multivariate_normal(scaled_mean, scaled_cov)
        
        # Apply the movement
        self.move(movement)


class GMMMixingObstacle(DynamicObstacle):
    """
    An obstacle that moves according to a mixture of Gaussians.
    Used to simulate multimodal movement patterns.
    """
    
    def __init__(self, position, radius, means, covariances, weights):
        """
        Initialize an obstacle that moves according to a GMM.
        
        Args:
            position: Initial position (3D vector)
            radius: Obstacle radius
            means: List of mean vectors for each Gaussian component
            covariances: List of covariance matrices for each component
            weights: Mixing weights for each component (must sum to 1)
        """
        super().__init__(position, radius)
        
        # Ensure weights sum to 1
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)
        
        self.n_components = len(weights)
        self.means = [np.array(mean, dtype=float) for mean in means]
        self.covariances = [np.array(cov, dtype=float) for cov in covariances]
        self.weights = weights
        
    def update(self, dt: float):
        """
        Update the obstacle position using a random sample from GMM.
        
        Args:
            dt: Time step
        """
        # Randomly select a component based on weights
        component_idx = np.random.choice(self.n_components, p=self.weights)
        
        # Get the selected component's parameters
        mean = self.means[component_idx] * dt
        cov = self.covariances[component_idx] * dt
        
        # Sample a random movement from this component
        movement = np.random.multivariate_normal(mean, cov)
        
        # Apply the movement
        self.move(movement)


class TimeVaryingGaussianObstacle(DynamicObstacle):
    """
    An obstacle with time-varying Gaussian parameters.
    Used to simulate changing movement patterns over time.
    """
    
    def __init__(self, position: np.ndarray, radius: float,
                 initial_mean: np.ndarray, initial_cov: np.ndarray,
                 variance_growth_rate: float = 0.0):
        """
        Initialize a time-varying Gaussian obstacle.
        
        Args:
            position: Initial position (3D vector)
            radius: Obstacle radius
            initial_mean: Initial mean vector for the Gaussian
            initial_cov: Initial covariance matrix
            variance_growth_rate: Rate at which variance increases over time
        """
        super().__init__(position, radius)
        self.mean = np.array(initial_mean, dtype=float)
        self.covariance = np.array(initial_cov, dtype=float)
        self.variance_growth_rate = variance_growth_rate
        self.time = 0.0  # Track elapsed time
        
    def update(self, dt: float):
        """
        Update the obstacle position with time-varying distribution.
        
        Args:
            dt: Time step
        """
        # Update elapsed time
        self.time += dt
        
        # Scale the covariance based on elapsed time
        current_cov = self.covariance + self.variance_growth_rate * self.time * np.eye(3)
        
        # Sample a random movement
        scaled_mean = self.mean * dt
        scaled_cov = current_cov * dt
        movement = np.random.multivariate_normal(scaled_mean, scaled_cov)
        
        # Apply the movement
        self.move(movement)


class ObstacleSet:
    """
    A collection of obstacles in the environment.
    Provides methods to update and check collisions for multiple obstacles.
    """
    
    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        self.obstacles = obstacles if obstacles is not None else []
        
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the collection."""
        self.obstacles.append(obstacle)
        
    def get_obstacles(self) -> List[Obstacle]:
        """Get the list of all obstacles."""
        return self.obstacles
        
    def update_all(self, dt: float):
        """Update all obstacles for one time step."""
        for obstacle in self.obstacles:
            obstacle.update(dt)
            
    def check_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """
        Check if a point collides with any obstacle.
        
        Args:
            point: 3D position to check
            safety_margin: Additional safety distance
            
        Returns:
            True if collision with any obstacle, False otherwise
        """
        return any(obs.is_collision(point, safety_margin) for obs in self.obstacles)
    
    def collect_movements(self) -> List[np.ndarray]:
        """
        Collect all movement vectors from all dynamic obstacles.
        Used for learning the distribution of movements.
        
        Returns:
            List of all movement vectors
        """
        movements = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, DynamicObstacle):
                movements.extend(obstacle.get_movements())
        return movements
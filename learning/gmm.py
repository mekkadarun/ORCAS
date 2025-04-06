import numpy as np
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple, Optional, Union


class GMMModel:
    """
    Gaussian Mixture Model wrapper for online learning of movement distributions.
    This replaces the Dirichlet Process Mixture Model from the paper with a
    more practical GMM implementation.
    """
    
    def __init__(self, n_components: int = 2, covariance_type: str = 'full', 
                 random_state: int = 0):
        """
        Initialize the GMM model.
        
        Args:
            n_components: Number of mixture components
            covariance_type: Type of covariance parameterization
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.is_fitted = False
        self.data = []
        
    def add_data(self, new_data: np.ndarray):
        """
        Add new movement data for online learning.
        
        Args:
            new_data: Array of new movement vectors to add
        """
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data)
            
        # Ensure 2D array
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
            
        if len(self.data) == 0:
            self.data = new_data
        else:
            self.data = np.vstack([self.data, new_data])
        
        # Model needs to be refitted
        self.is_fitted = False
        
    def fit(self):
        """Fit the GMM to the current data."""
        if len(self.data) < self.n_components:
            # Not enough data points yet
            return False
            
        self.model.fit(self.data)
        self.is_fitted = True
        return True
    
    # In gmm.py, add to the GMMModel class
    def update_with_sliding_window(self, new_data: np.ndarray, window_size: int = 50):
        """
        Update the GMM using a sliding window of the most recent observations.
        
        Args:
            new_data: New observation data
            window_size: Maximum window size
        """
        # Add new data
        self.add_data(new_data)
        
        # Keep only the most recent data points
        if len(self.data) > window_size:
            self.data = self.data[-window_size:]
        
        # Refit the model
        self.fit()
        
        return self.is_fitted
        
    def get_mixture_components(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Get the mixture components (weights, means, covariances).
        
        Returns:
            Tuple of (weights, means, covariances)
        """
        if not self.is_fitted:
            self.fit()
            
        if not self.is_fitted:
            # Still not fitted (insufficient data)
            return np.array([1.0]), [np.zeros(self.data.shape[1])], [np.eye(self.data.shape[1])]
            
        return self.model.weights_, self.model.means_, self.model.covariances_
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the mixture model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples
        """
        if not self.is_fitted:
            self.fit()
            
        if not self.is_fitted:
            # Not fitted yet, return zeros
            dim = self.data.shape[1] if len(self.data) > 0 else 3
            return np.zeros((n_samples, dim))
            
        return self.model.sample(n_samples)[0]


class AmbiguitySet:
    """
    Implements the data-stream-driven ambiguity set construction
    as described in the paper, using GMM instead of DPMM.
    """
    
    def __init__(self, dimension: int = 3, support_bounds: Optional[np.ndarray] = None):
        """
        Initialize the ambiguity set.
        
        Args:
            dimension: Dimension of the space (typically 3 for 3D)
            support_bounds: Optional bounds for the support set
        """
        self.dimension = dimension
        self.support_bounds = support_bounds if support_bounds is not None else np.ones(dimension) * 10
        self.gmm_model = GMMModel(n_components=2)  # Default to 2 components
        
    def add_movement_data(self, movement_data: np.ndarray):
        """
        Add new movement data to update the ambiguity set.
        
        Args:
            movement_data: Array of movement vectors
        """
        self.gmm_model.add_data(movement_data)
        
    def update(self):
        """Update the ambiguity set by refitting the GMM."""
        return self.gmm_model.fit()
        
    def get_components(self) -> List[Dict]:
        """
        Get the components of the ambiguity set.
        
        Returns:
            List of dictionaries with component parameters
        """
        weights, means, covariances = self.gmm_model.get_mixture_components()
        
        components = []
        for i in range(len(weights)):
            components.append({
                'weight': weights[i],
                'mean': means[i],
                'covariance': covariances[i]
            })
            
        return components
    
    def get_support_set(self) -> Dict:
        """
        Get the support set for the ambiguity set.
        
        Returns:
            Dictionary describing the support set H
        """
        return {
            'type': 'box',
            'bounds': self.support_bounds
        }


class OnlineDistributionLearner:
    """
    Manages the online learning of movement distributions
    for multiple obstacles.
    """
    
    def __init__(self):
        """Initialize the online distribution learner."""
        self.obstacle_ambiguity_sets = {}
        
    def create_ambiguity_set(self, obstacle_id: Union[str, int], 
                            dimension: int = 3,
                            support_bounds: Optional[np.ndarray] = None) -> AmbiguitySet:
        """
        Create a new ambiguity set for an obstacle.
        
        Args:
            obstacle_id: Identifier for the obstacle
            dimension: Dimension of the space
            support_bounds: Optional bounds for the support set
            
        Returns:
            The created ambiguity set
        """
        ambiguity_set = AmbiguitySet(dimension, support_bounds)
        self.obstacle_ambiguity_sets[obstacle_id] = ambiguity_set
        return ambiguity_set
    
    def get_ambiguity_set(self, obstacle_id: Union[str, int]) -> Optional[AmbiguitySet]:
        """
        Get the ambiguity set for a specific obstacle.
        
        Args:
            obstacle_id: Identifier for the obstacle
            
        Returns:
            The ambiguity set if it exists, None otherwise
        """
        return self.obstacle_ambiguity_sets.get(obstacle_id)
    
    # In gmm.py, modify the add_movement_data method in OnlineDistributionLearner
    def add_movement_data(self, obstacle_id: Union[str, int], movement_data: np.ndarray, 
                        window_size: int = 50):
        """
        Add movement data for an obstacle and update with sliding window.
        
        Args:
            obstacle_id: Identifier for the obstacle
            movement_data: Array of movement vectors
            window_size: Size of the sliding window
        """
        ambiguity_set = self.get_ambiguity_set(obstacle_id)
        
        if ambiguity_set is None:
            ambiguity_set = self.create_ambiguity_set(obstacle_id)
            
        # Add data with sliding window
        ambiguity_set.gmm_model.update_with_sliding_window(movement_data, window_size)
    
    def update_all(self) -> Dict[Union[str, int], bool]:
        """
        Update all ambiguity sets.
        
        Returns:
            Dictionary mapping obstacle IDs to update success status
        """
        results = {}
        for obstacle_id, ambiguity_set in self.obstacle_ambiguity_sets.items():
            results[obstacle_id] = ambiguity_set.update()
        return results
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMObstacle:
    def __init__(self, position, radius=0.5, n_components=3, random_state=42):
        """
        Initialize a Gaussian Mixture Model-based obstacle
        
        Args:
            position: Initial position of the obstacle (x, y)
            radius: Radius of the obstacle
            n_components: Number of Gaussian components in the mixture model
            random_state: Random seed for reproducibility
        """
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize GMM model
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state
        )
        
        # Store movement history for fitting GMM
        self.movement_history = []
        
        # Placeholder for latest sampled movement
        self.latest_movement = np.zeros(2)
        
    def get_position(self):
        """Return current position of the obstacle"""
        return self.position.copy()
    
    def update_position(self, dt, use_gmm=True):
        """
        Update position based on GMM prediction or noise if not enough data
        
        Args:
            dt: Time step
            use_gmm: Whether to use GMM for movement prediction
        """
        if use_gmm and len(self.movement_history) >= 10:
            # Sample movement from the GMM
            movement = self._sample_from_gmm() * dt
            self.latest_movement = movement
        else:
            # Use random noise until enough data is collected
            movement = np.random.normal(0, 0.1, size=2) * dt
            self.latest_movement = movement
            
        # Update position
        self.position += movement
        
        # Record this movement for future GMM fitting
        self.movement_history.append(movement / dt)  # Store normalized movement
        
        # Refit GMM when we have enough new data points
        if len(self.movement_history) % 5 == 0 and len(self.movement_history) >= 10:
            self._fit_gmm()
    
    def _fit_gmm(self):
        """Fit GMM to movement history with stability improvements"""
        if len(self.movement_history) < 10:
            return
            
        # Convert movement history to numpy array
        movement_data = np.array(self.movement_history[-50:])  # Use more data points for better modeling
        
        # Handle degenerate cases
        if np.all(np.std(movement_data, axis=0) < 1e-6):
            # If data is almost constant, use simple model
            self.gmm = GaussianMixture(
                n_components=1,
                covariance_type='diag',
                random_state=self.random_state
            )
            self.gmm.means_ = np.array([np.mean(movement_data, axis=0)])
            self.gmm.covariances_ = np.array([np.diag([max(0.01, np.var(movement_data[:, 0])), 
                                                       max(0.01, np.var(movement_data[:, 1]))])])
            self.gmm.weights_ = np.array([1.0])
            return
        
        try:
            # Try to fit GMM with increasing regularization until successful
            regularization = 1e-6
            max_attempts = 5
            
            for attempt in range(max_attempts):
                try:
                    # Initialize with k-means to improve stability
                    self.gmm = GaussianMixture(
                        n_components=self.n_components,
                        covariance_type='full',
                        random_state=self.random_state,
                        reg_covar=regularization,
                        init_params='kmeans',
                        max_iter=300
                    )
                    self.gmm.fit(movement_data)
                    
                    # Check if fit was successful
                    if np.any(np.isnan(self.gmm.means_)) or np.any(np.isnan(self.gmm.covariances_)):
                        raise ValueError("NaN values in fitted GMM")
                        
                    # Check for small weights and merge components if needed
                    small_idx = np.where(self.gmm.weights_ < 0.05)[0]
                    if len(small_idx) > 0:
                        self._merge_small_components(small_idx)
                        
                    return
                except Exception as e:
                    regularization *= 10
                    print(f"GMM fitting attempt {attempt+1} failed, increasing regularization to {regularization}")
                    
            # If all attempts failed, use a simple single-component model
            self.gmm = GaussianMixture(
                n_components=1,
                covariance_type='full',
                random_state=self.random_state
            )
            self.gmm.means_ = np.array([np.mean(movement_data, axis=0)])
            self.gmm.covariances_ = np.array([np.cov(movement_data.T) + np.eye(2) * 0.01])
            self.gmm.weights_ = np.array([1.0])
            
        except Exception as e:
            print(f"GMM fitting failed: {e}, using fallback model")
            # Fallback to a simple model
            self.gmm = GaussianMixture(
                n_components=1,
                covariance_type='full',
                random_state=self.random_state
            )
            self.gmm.means_ = np.array([np.mean(movement_data, axis=0)])
            self.gmm.covariances_ = np.array([np.eye(2) * 0.01])
            self.gmm.weights_ = np.array([1.0])
    
    def _merge_small_components(self, small_idx):
        """Merge small weight components into larger ones"""
        if len(small_idx) == 0:
            return
            
        # Keep only significant components
        keep_idx = np.where(self.gmm.weights_ >= 0.05)[0]
        if len(keep_idx) == 0:
            # If all weights are small, keep the largest one
            keep_idx = [np.argmax(self.gmm.weights_)]
            
        # Renormalize weights
        new_weights = self.gmm.weights_[keep_idx]
        new_weights = new_weights / np.sum(new_weights)
        
        # Update model parameters
        self.gmm.means_ = self.gmm.means_[keep_idx]
        self.gmm.covariances_ = self.gmm.covariances_[keep_idx]
        self.gmm.weights_ = new_weights
        
        # Update number of components
        self.n_components = len(keep_idx)
    
    def _sample_from_gmm(self):
        """Sample a movement vector from the GMM"""
        sample = self.gmm.sample(1)[0][0]
        return sample
    
    def get_uncertainty_ellipses(self, time_horizon=10, time_step=0.1):
        """
        Get prediction ellipses for a time horizon
        
        Args:
            time_horizon: Number of time steps to predict
            time_step: Size of each time step
            
        Returns:
            List of dictionaries containing mean and covariance for each component
            at each time step
        """
        if len(self.movement_history) < 10:
            # Return a simple circular uncertainty if not enough data
            uncertainty = []
            for t in range(1, time_horizon + 1):
                # Increasing radius with time
                uncertainty.append({
                    'means': [self.position + self.latest_movement * t * time_step], 
                    'covariances': [np.eye(2) * (0.1 * t * time_step)**2],
                    'weights': [1.0]
                })
            return uncertainty
        
        # Get components from the GMM
        means = self.gmm.means_
        covariances = self.gmm.covariances_
        weights = self.gmm.weights_
        
        # Create uncertainty ellipses for each time step
        uncertainty = []
        for t in range(1, time_horizon + 1):
            # Compute predicted means by adding scaled movement trends to current position
            # This keeps means relative to the current position
            scaled_means = []
            
            for mean in means:
                # Mean represents likely movement direction/speed
                # Scale it by time and add to current position
                prediction = self.position + mean * t * time_step
                scaled_means.append(prediction)
            
            # Scale covariances with time (uncertainty grows with prediction time)
            # Use a lower scaling factor to reduce conservatism
            time_scale = np.sqrt(t * time_step)  # Square root scaling instead of quadratic
            scaled_covs = [cov * time_scale for cov in covariances]
            
            uncertainty.append({
                'means': scaled_means,
                'covariances': scaled_covs,
                'weights': weights
            })
            
        return uncertainty
    
    def get_constraint_parameters(self):
        """
        Return parameters needed for collision avoidance constraint
        
        Returns:
            Dictionary with position, radius, and uncertainty parameters
        """
        return {
            'center': self.position,
            'radius': self.radius,
            'uncertainty': self.get_uncertainty_ellipses()
        }
    
    def distance_to(self, point):
        """
        Calculate distance from point to obstacle surface
        
        Args:
            point: Point to calculate distance from
            
        Returns:
            Distance from point to obstacle surface (negative if inside)
        """
        return np.linalg.norm(point - self.position) - self.radius
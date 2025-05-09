import numpy as np
import sklearn.mixture as mix
from control.real_ambiguity_set import AmbiguitySet

class GMMObstacle:
    """Obstacle with Gaussian Mixture Model motion prediction with full 3D support"""
    
    def __init__(self, initial_position, radius=0.5, n_components=3, is_3d=False, confidence_level=0.95):
        """Initialize an obstacle with GMM-based movement"""
        self.position = np.array(initial_position, dtype=float)
        self.radius = radius
        self.n_components = n_components
        self.is_3d = is_3d
        self.confidence_level = confidence_level
        
        # Movement history for GMM learning
        self.movement_history = []
        
        # Current GMM model for predicting movement
        self.gmm = None
        
        # Reference to previous movement for calculating velocity
        self.latest_movement = None
        
        # Create ambiguity set for robust motion control
        self.ambiguity_set = AmbiguitySet(
            max_components=n_components,
            confidence_level=confidence_level
        )
        
    def set_confidence_level(self, confidence_level):
        """Update confidence level for this obstacle and its ambiguity set"""
        if self.confidence_level != confidence_level:
            self.confidence_level = confidence_level
            # Update ambiguity set confidence level
            self.ambiguity_set.set_confidence_level(confidence_level)
            return True
        return False
        
    def _fit_gmm(self):
        """Fit GMM to movement history data"""
        if len(self.movement_history) < 3:
            return False
            
        data = np.array(self.movement_history)
        
        try:
            self.gmm = mix.GaussianMixture(
                n_components=min(self.n_components, len(data) // 2),
                covariance_type='full',
                random_state=42,
                reg_covar=1e-6
            )
            self.gmm.fit(data)
            return True
        except Exception as e:
            print(f"Error fitting GMM: {e}")
            return False
    
    def update_position(self, dt, use_gmm=True, maintain_z=False, goal_z=None):
        """Update obstacle position based on learned movement patterns"""
        if use_gmm and self.gmm is not None and len(self.movement_history) >= 3:
            # Sample from GMM with small noise
            movement = self.gmm.sample(1)[0][0]
            
            # Add small random noise for variety
            if self.is_3d:
                noise = np.random.normal(0, 0.01, 3)
            else:
                noise = np.random.normal(0, 0.01, 2)
                
            movement = movement + noise
        else:
            # Default movement pattern if GMM isn't available
            if self.is_3d:
                movement = np.random.normal(0, 0.1, 3)
            else:
                movement = np.random.normal(0, 0.1, 2)
        
        # If maintain_z flag is set and we have a goal_z value, restrict z movement
        if maintain_z and goal_z is not None and self.is_3d:
            # Force z-component to maintain altitude with minimal movement
            if len(movement) >= 3:
                # Calculate required z-movement to return to goal_z (with damping)
                z_error = goal_z - self.position[2]
                z_correction = 0.3 * z_error  # Damped correction
                
                # Set z-component of movement to maintain altitude
                movement[2] = z_correction
        
        # Scale movement by dt
        movement = movement * dt
        
        # Update position
        self.position = self.position + movement
        
        # Store movement for history
        self.latest_movement = movement
        self.movement_history.append(movement / dt)  # Store velocity, not displacement
        
        # Limit history length
        if len(self.movement_history) > 100:
            self.movement_history.pop(0)
            
        # Periodically refit GMM and update ambiguity set
        if len(self.movement_history) % 5 == 0:
            self._fit_gmm()
            # Make sure ambiguity set is also updated
            self.ambiguity_set.add_movement_data(movement / dt)
            self.ambiguity_set.update_mixture_model()
            
        return movement
        
    def get_position(self):
        """Get current position of the obstacle"""
        return self.position.copy()
        
    def get_constraint_parameters(self):
        """Get parameters needed for creating constraints in MPC"""
        params = {
            'center': self.position.copy(),
            'radius': self.radius,
            'uncertainty': []
        }
        
        # Add uncertainty from ambiguity set if available
        if hasattr(self, 'ambiguity_set'):
            # Ensure ambiguity set uses the current confidence level
            self.ambiguity_set.set_confidence_level(self.confidence_level)
            
            for t in range(5):  # Look ahead 5 steps
                uncertainty = self.ambiguity_set.get_uncertainty(t)
                if uncertainty:
                    # Handle dimensionality expansion for 3D
                    if self.is_3d and uncertainty['means'] and len(uncertainty['means'][0]) < 3:
                        # Expand 2D means to 3D
                        for i in range(len(uncertainty['means'])):
                            uncertainty['means'][i] = np.append(uncertainty['means'][i], 0)
                            
                        # Expand 2D covariances to 3D
                        for i in range(len(uncertainty['covariances'])):
                            cov_2d = uncertainty['covariances'][i]
                            cov_3d = np.zeros((3, 3))
                            cov_3d[:2, :2] = cov_2d
                            cov_3d[2, 2] = 0.01  # Small variance in z
                            uncertainty['covariances'][i] = cov_3d
                    
                    # Add to parameters
                    params['uncertainty'].append(uncertainty)
        
        return params
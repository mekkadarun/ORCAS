import numpy as np
import sklearn.mixture as mix
from scipy.stats import chi2

class AmbiguitySet:
    """Implementation of a data-stream-driven ambiguity set for distributionally robust collision avoidance with moving obstacles"""
    
    def __init__(self, max_components=5, confidence_level=0.90, regularization=1e-6):
        self.max_components = max_components
        self.confidence_level = confidence_level
        self.regularization = regularization
        self.chi2_val = chi2.ppf(confidence_level, df=3)  # Default to 3D
        self.movement_history = []
        self.mixture_model = None
        self.ambiguity_params = None
        
        # Parameters for the DPMM ambiguity set
        self.basic_ambiguity_sets = []
        self.gamma_weights = []  # Mixing weights
    
    def set_confidence_level(self, confidence_level):
        """Update confidence level and related parameters"""
        if self.confidence_level != confidence_level:
            self.confidence_level = confidence_level
            # Update chi2 value
            dims = 3  # Default to 3D
            if self.movement_history and len(self.movement_history) > 0:
                dims = len(self.movement_history[0])
            self.chi2_val = chi2.ppf(confidence_level, df=dims)
            # Update ambiguity set if we already have a model
            if self.mixture_model is not None:
                self.update_ambiguity_set()
            return True
        return False
    
    def add_movement_data(self, movement):
        """Add new movement observation to history."""
        self.movement_history.append(movement)
    
    def update_mixture_model(self):
        """Fit a mixture model to movement history implementing the online learning approach."""
        if len(self.movement_history) < 3:
            return False
            
        data = np.array(self.movement_history)
        
        # Check if data is 2D or 3D
        is_3d = data.shape[1] >= 3
        
        # Find optimal number of components using BIC
        best_bic = np.inf
        best_model = None
        
        # Try different component counts to find optimal model
        for n_components in range(1, min(self.max_components + 1, len(data) // 2 + 1)):
            try:
                gmm = mix.GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    reg_covar=self.regularization
                )
                gmm.fit(data)
                bic = gmm.bic(data)
                
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
            except Exception as e:
                print(f"Error fitting GMM with {n_components} components: {e}")
                continue
                
        # Fallback to single component if fitting fails
        if best_model is None:
            try:
                gmm = mix.GaussianMixture(
                    n_components=1,
                    covariance_type='full',
                    random_state=42,
                    reg_covar=self.regularization * 10  # Stronger regularization
                )
                gmm.fit(data)
                best_model = gmm
            except Exception as e:
                print(f"Fallback GMM fitting failed: {e}")
                return False
            
        self.mixture_model = best_model
        
        # Update chi2 value based on data dimensionality
        self.chi2_val = chi2.ppf(self.confidence_level, df=data.shape[1])
        
        self.update_ambiguity_set()
        
        return True
    
    def update_ambiguity_set(self):
        """Construct the ambiguity set based on the fitted mixture model."""
        if self.mixture_model is None:
            return False
        
        # Extract GMM parameters
        weights = self.mixture_model.weights_
        means = self.mixture_model.means_
        covs = self.mixture_model.covariances_
        
        # Regularize covariances to ensure positive definiteness
        reg_covs = []
        for cov in covs:
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < self.regularization):
                # Add regularization based on smallest eigenvalue
                min_eig = max(0, np.min(eigvals))
                reg_factor = self.regularization - min_eig
                reg_covs.append(cov + np.eye(cov.shape[0]) * reg_factor)
            else:
                reg_covs.append(cov)
        
        # Update mixing weights (γ in the paper)
        self.gamma_weights = weights
        
        # Construct the basic ambiguity sets
        self.basic_ambiguity_sets = []
        for j in range(len(weights)):
            if weights[j] < 0.05:  # Skip small weight components
                continue
                
            # Each basic ambiguity set defined by mean and covariance
            basic_set = {
                'mean': means[j],
                'covariance': reg_covs[j],
                'weight': weights[j]
            }
            self.basic_ambiguity_sets.append(basic_set)
        
        # Store in ambiguity parameters format
        self.ambiguity_params = {
            'n_components': len(self.basic_ambiguity_sets),
            'weights': [set['weight'] for set in self.basic_ambiguity_sets],
            'means': [set['mean'] for set in self.basic_ambiguity_sets],
            'covariances': [set['covariance'] for set in self.basic_ambiguity_sets]
        }
        
        return True
    
    def get_uncertainty(self, time_index=0):
        """Get uncertainty parameters for a specific prediction time."""
        if self.ambiguity_params is None:
            return None
        
        # Scale uncertainties based on prediction time and confidence level
        # Higher confidence means more uncertainty growth over time
        confidence_factor = 1.0 + (self.confidence_level - 0.9)
        time_scale = 1.0 + 0.2 * time_index * confidence_factor
        
        uncertainty = {
            'means': [],
            'covariances': [],
            'weights': []
        }
        
        for i in range(self.ambiguity_params['n_components']):
            if self.ambiguity_params['weights'][i] < 0.05:
                continue
                
            # Predicted mean based on movement trend
            uncertainty['means'].append(
                self.ambiguity_params['means'][i] * time_scale
            )
            
            # Scale covariance with time to increase future uncertainty
            # Higher confidence means larger uncertainty growth
            uncertainty['covariances'].append(
                self.ambiguity_params['covariances'][i] * (time_scale**2)
            )
            
            uncertainty['weights'].append(self.ambiguity_params['weights'][i])
        
        # Normalize weights
        if uncertainty['weights']:
            total_weight = sum(uncertainty['weights'])
            if total_weight > 0:
                uncertainty['weights'] = [w/total_weight for w in uncertainty['weights']]
        
        return uncertainty
    
    def get_ambiguity_set_parameters(self):
        """Returns the parameters of the ambiguity set as defined in equation (10) in the paper."""
        if not self.basic_ambiguity_sets:
            return None
            
        return {
            'gamma_weights': [set['weight'] for set in self.basic_ambiguity_sets],
            'means': [set['mean'] for set in self.basic_ambiguity_sets],
            'covariances': [set['covariance'] for set in self.basic_ambiguity_sets]
        }
    
    def worst_case_risk(self, position, obstacle_pos, min_dist):
        """Calculate worst-case risk based on the ambiguity set with proper 3D support."""
        if self.ambiguity_params is None:
            dist = np.linalg.norm(position - obstacle_pos)
            if dist <= min_dist:
                return 1.0
            return (min_dist / dist)**2
        
        # Base risk calculation
        dist = np.linalg.norm(position - obstacle_pos)
        if dist <= min_dist:
            return 1.0
        
        # Base risk increases with confidence level
        confidence_factor = 1.0 + (self.confidence_level - 0.9) * 3.0
        base_risk = min(1.0, (min_dist / dist)**(2.0 / confidence_factor))
        total_risk = base_risk
        
        # Match dimensions for position and obstacle position
        pos_dim = len(position)
        obs_dim = len(obstacle_pos)
        
        # Add risk from each mixture component
        for i in range(self.ambiguity_params['n_components']):
            weight = self.ambiguity_params['weights'][i]
            if weight < 0.05:
                continue
                
            mean = self.ambiguity_params['means'][i]
            cov = self.ambiguity_params['covariances'][i]
            
            # Handle dimensionality matching
            mean_dim = len(mean)
            if pos_dim != mean_dim or obs_dim != mean_dim:
                # Match to the lowest dimension
                min_dim = min(pos_dim, obs_dim, mean_dim)
                position_adj = position[:min_dim]
                obstacle_pos_adj = obstacle_pos[:min_dim]
                mean_adj = mean[:min_dim]
                cov_adj = cov[:min_dim, :min_dim]
            else:
                position_adj = position
                obstacle_pos_adj = obstacle_pos
                mean_adj = mean
                cov_adj = cov
            
            # Predicted position of the obstacle
            pred_obstacle_pos = obstacle_pos_adj + mean_adj
            
            # Direction from predicted obstacle position to robot
            direction = position_adj - pred_obstacle_pos
            norm_dir = np.linalg.norm(direction)
            
            if norm_dir < 1e-6:
                direction = np.ones(len(direction)) / np.sqrt(len(direction))
                norm_dir = 1.0
            else:
                direction = direction / norm_dir
            
            # Variance along direction
            variance = direction.T @ cov_adj @ direction
            
            # Scale factor based on chi-square value - higher confidence means higher scale
            scale = np.sqrt(self.chi2_val * variance)
            
            # Compute distance to predicted obstacle position
            pred_dist = np.linalg.norm(position_adj - pred_obstacle_pos)
            
            # Component risk (using exponential decay model)
            # Higher chi2_val (confidence) means SLOWER decay with distance
            confidence_scale = max(1.0, self.chi2_val / 3.0)
            comp_risk = weight * np.exp(-(pred_dist - min_dist)**2 / (2 * scale**2 * confidence_scale))
            
            # Combine risks (avoiding double-counting)
            total_risk = total_risk + comp_risk * (1.0 - total_risk)
        
        return min(total_risk, 1.0)
    
    def sample_movement(self):
        """Sample a movement from the current mixture model."""
        if self.mixture_model is None:
            # Default random movement
            if len(self.movement_history) > 0 and len(self.movement_history[0]) >= 3:
                return np.random.normal(0, 0.1, 3)  # 3D case
            else:
                return np.random.normal(0, 0.1, 2)  # 2D case
        
        return self.mixture_model.sample(1)[0][0]
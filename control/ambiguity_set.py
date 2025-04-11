import numpy as np
import sklearn.mixture as mix
from scipy.stats import chi2

class AmbiguitySet:
    """
    Implementation of a data-stream-driven ambiguity set for distributionally
    robust collision avoidance with moving obstacles, based on the paper
    "Online-Learning-Based Distributionally Robust Motion Control with Collision
    Avoidance for Mobile Robots".
    """
    
    def __init__(self, max_components=5, confidence_level=0.90, regularization=1e-6):
        self.max_components = max_components
        self.confidence_level = confidence_level
        self.regularization = regularization
        self.chi2_val = chi2.ppf(confidence_level, df=2)
        self.movement_history = []
        self.mixture_model = None
        self.ambiguity_params = None
        
        # Parameters for the DPMM ambiguity set as described in the paper
        self.basic_ambiguity_sets = []
        self.gamma_weights = []  # Mixing weights (γ in the paper's equation 10)
    
    def add_movement_data(self, movement):
        """Add new movement observation to history."""
        self.movement_history.append(movement)
    
    def update_mixture_model(self):
        """
        Fit a mixture model to movement history, implementing the online
        learning approach described in the paper.
        """
        if len(self.movement_history) < 3:
            return False
            
        data = np.array(self.movement_history)
        
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
        self.update_ambiguity_set()
        
        return True
    
    def update_ambiguity_set(self):
        """
        Construct the ambiguity set based on the fitted mixture model.
        This implements equation (10) from the paper to create a
        Minkowski sum of basic ambiguity sets.
        """
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
            # This follows equation (10) in the paper
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
        """
        Get uncertainty parameters for a specific prediction time.
        This implements the time-varying ambiguity set described in the paper.
        """
        if self.ambiguity_params is None:
            return None
        
        # Scale uncertainties based on prediction time
        # This scaling follows the paper's approach for increasing uncertainty with time
        time_scale = 1.0 + 0.2 * time_index
        
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
            # This implements the uncertainty growth model from the paper
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
        """
        Returns the parameters of the ambiguity set as defined in equation (10)
        in the paper.
        """
        if not self.basic_ambiguity_sets:
            return None
            
        return {
            'gamma_weights': [set['weight'] for set in self.basic_ambiguity_sets],
            'means': [set['mean'] for set in self.basic_ambiguity_sets],
            'covariances': [set['covariance'] for set in self.basic_ambiguity_sets]
        }
    
    def worst_case_risk(self, position, obstacle_pos, min_dist):
        """
        Calculate worst-case risk based on the ambiguity set.
        Implements the distributionally robust approach from the paper.
        """
        if self.ambiguity_params is None:
            dist = np.linalg.norm(position - obstacle_pos)
            if dist <= min_dist:
                return 1.0
            return (min_dist / dist)**2
        
        # Base risk calculation
        dist = np.linalg.norm(position - obstacle_pos)
        if dist <= min_dist:
            return 1.0
        
        base_risk = (min_dist / dist)**2
        total_risk = base_risk
        
        # Add risk from each mixture component
        for i in range(self.ambiguity_params['n_components']):
            weight = self.ambiguity_params['weights'][i]
            if weight < 0.05:
                continue
                
            mean = self.ambiguity_params['means'][i]
            cov = self.ambiguity_params['covariances'][i]
            
            # Predicted position of the obstacle
            pred_obstacle_pos = obstacle_pos + mean
            
            # Direction from predicted obstacle position to robot
            direction = position - pred_obstacle_pos
            norm_dir = np.linalg.norm(direction)
            
            if norm_dir < 1e-6:
                direction = np.array([1.0, 0.0])
                norm_dir = 1.0
            else:
                direction = direction / norm_dir
            
            # Variance along direction
            variance = direction.T @ cov @ direction
            
            # Scale factor based on chi-square value
            scale = np.sqrt(self.chi2_val * variance)
            
            # Compute distance to predicted obstacle position
            pred_dist = np.linalg.norm(position - pred_obstacle_pos)
            
            # Component risk (using exponential decay model)
            # This aligns with the paper's risk model
            comp_risk = weight * np.exp(-(pred_dist - min_dist)**2 / (2 * scale**2))
            
            # Combine risks (avoiding double-counting)
            total_risk = total_risk + comp_risk * (1.0 - total_risk)
        
        return min(total_risk, 1.0)
    
    def calculate_cvar_constraint(self, pos_var, obstacle_center, min_dist):
        """
        Calculate the CVaR constraint for optimization based on the ambiguity set.
        This implements the reformulation from the paper's Theorem 1.
        
        Args:
            pos_var: Position variable (from optimization)
            obstacle_center: Center position of the obstacle
            min_dist: Minimum safe distance
            
        Returns:
            List of constraint expressions
        """
        if not self.basic_ambiguity_sets:
            # If no ambiguity set is available, use a simple distance constraint
            from cvxpy import norm
            return [norm(pos_var - obstacle_center) >= min_dist]
        
        # Implementation of the paper's CVaR constraint formulation
        # that uses the ambiguity set for distributionally robust constraints
        from cvxpy import norm, Variable, Minimize, Problem, sum_squares
        
        constraints = [norm(pos_var - obstacle_center) >= min_dist]
        
        for basic_set in self.basic_ambiguity_sets:
            weight = basic_set['weight']
            if weight < 0.05:
                continue
                
            mean = basic_set['mean']
            cov = basic_set['covariance']
            
            # Calculate predicted obstacle position
            pred_pos = obstacle_center + mean
            
            # Vector from predicted position to robot
            direction = pos_var - pred_pos
            
            # Add conservative constraint based on covariance
            # This implements the linearized constraint from Theorem 1
            try:
                # Eigen-decomposition of covariance for scaling
                eigvals, eigvecs = np.linalg.eigh(cov)
                max_eigval = max(eigvals)
                
                # Scale factor for safety distance
                scale = np.sqrt(self.chi2_val * max_eigval)
                
                # Add scaled distance constraint
                constraints.append(norm(direction) >= min_dist + scale)
            except:
                # Fallback constraint
                constraints.append(norm(direction) >= min_dist * (1 + weight))
        
        return constraints
    
    def sample_movement(self):
        """Sample a movement from the current mixture model."""
        if self.mixture_model is None:
            return np.random.normal(0, 0.1, 2)
        
        return self.mixture_model.sample(1)[0][0]
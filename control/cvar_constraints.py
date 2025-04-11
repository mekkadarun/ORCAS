import numpy as np
import cvxpy as cvx
from scipy.stats import chi2

class CVaRObstacleAvoidance:
    def __init__(self, confidence_level=0.97, safety_margin=0.6):
        self.confidence_level = confidence_level
        self.safety_margin = safety_margin
        self.chi2_val = chi2.ppf(confidence_level, df=2)
    
    def calculate_cvar_constraint(self, pos_var, obstacle, t, min_dist):
        """
        Calculate collision avoidance constraints using distributionally robust approach.
        Implements the key formulation from the paper's Theorem 1.
        
        Args:
            pos_var: Position variable (from optimization)
            obstacle: Obstacle object with ambiguity set
            t: Time step index
            min_dist: Minimum safe distance
            
        Returns:
            List of constraint expressions
        """
        constraints = []
        params = obstacle.get_constraint_parameters()
        center = params['center']
        
        # Base constraint for current obstacle position
        constraints.append(
            cvx.sum_squares(pos_var - center) >= min_dist**2
        )
        
        # Early return if no uncertainty data
        uncertainty_idx = min(t-1, len(params['uncertainty'])-1)
        if uncertainty_idx < 0:
            return constraints
        
        # Get uncertainty for current prediction time
        uncertainty = params['uncertainty'][uncertainty_idx]
        means = uncertainty['means']
        covariances = uncertainty['covariances']
        weights = uncertainty['weights']
        
        # Apply constraints for each component in the ambiguity set
        for j, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
            if weight < 0.05:  # Skip negligible components
                continue
                
            # Regularize covariance for numerical stability
            reg_cov = cov + np.eye(2) * 1e-6
            
            try:
                # Current position of obstacle
                curr_pos = np.array(center)
                
                # Vector from predicted mean to position
                vec_to_robot = curr_pos - mean
                
                # Handle special case when vector is very small
                if np.linalg.norm(vec_to_robot) < 1e-6:
                    # Use squared distance constraint
                    weighted_min_dist = min_dist * (1.0 + weight)
                    constraints.append(
                        cvx.sum_squares(pos_var - mean) >= weighted_min_dist**2
                    )
                    continue
                
                # Normalize to get direction
                direction = vec_to_robot / np.linalg.norm(vec_to_robot)
                
                # Calculate scale factor based on chi-square distribution
                # This implements the paper's formulation for CVaR constraint
                scale_factor = np.sqrt(self.chi2_val * direction.T @ reg_cov @ direction)
                
                # Adjust minimum distance based on component weight
                weighted_min_dist = min_dist * (1.0 + weight)
                
                # Add linearized constraint (DCP-compliant)
                constraints.append(
                    direction @ (pos_var - mean) >= weighted_min_dist + scale_factor
                )
                
            except Exception as e:
                print(f"Constraint generation failed: {e}")
                # Use fallback constraint
                weighted_min_dist = min_dist * (1.0 + weight)
                constraints.append(
                    cvx.sum_squares(pos_var - mean) >= weighted_min_dist**2
                )
        
        return constraints
    
    def calculate_risk(self, position, obstacle):
        """
        Calculate distributionally robust risk of collision with an obstacle.
        Implements the risk assessment approach from the paper.
        
        Args:
            position: Current or test position
            obstacle: Obstacle with ambiguity set
            
        Returns:
            Risk value between 0 and 1
        """
        params = obstacle.get_constraint_parameters()
        center = params['center']
        risk = 0.0
        
        # Base risk from current position
        dist = np.linalg.norm(position - center)
        radius = params['radius']
        min_safe_dist = radius + self.safety_margin
        
        if dist <= min_safe_dist:
            risk = 1.0
        else:
            # Risk decreases with distance according to paper's model
            risk = (min_safe_dist / dist) ** 2
            
        # Add risk from ambiguity set components
        if 'uncertainty' in params and len(params['uncertainty']) > 0:
            uncertainty = params['uncertainty'][0]  # First time step prediction
            
            # Consider each mixture component
            for mean, cov, weight in zip(uncertainty['means'], 
                                         uncertainty['covariances'],
                                         uncertainty['weights']):
                if weight < 0.05:  # Skip negligible components
                    continue
                
                # Distance to component mean
                comp_dist = np.linalg.norm(position - mean)
                
                try:
                    # Calculate risk based on uncertainty model
                    eigvals = np.linalg.eigvalsh(cov)
                    max_std = np.sqrt(max(eigvals))
                    
                    # Exponential risk model from paper
                    comp_risk = weight * np.exp(-(comp_dist**2) / (2 * max_std**2 * self.chi2_val))
                    
                    # Combine risks with proper weighting
                    risk = risk + comp_risk * (1.0 - risk)
                    
                except np.linalg.LinAlgError:
                    # Fallback risk calculation
                    comp_risk = weight * (min_safe_dist / max(comp_dist, min_safe_dist)) ** 2
                    risk = max(risk, comp_risk)
        
        return min(risk, 1.0)  # Ensure risk is bounded
    
    def calculate_total_risk(self, position, obstacles):
        """
        Calculate total risk from all obstacles using distributionally robust approach.
        
        Args:
            position: Current or test position
            obstacles: List of obstacles
            
        Returns:
            Combined risk value between 0 and 1
        """
        total_risk = 0.0
        
        # Calculate risk from each obstacle
        for obs in obstacles:
            obs_risk = self.calculate_risk(position, obs)
            
            # Combine risks using method described in paper
            total_risk = total_risk + obs_risk * (1.0 - total_risk)
            
        return total_risk
    
    def generate_risk_map(self, x_range, y_range, resolution, obstacles):
        """
        Generate a 2D risk map for visualization.
        
        Args:
            x_range: Range of x values (min, max)
            y_range: Range of y values (min, max)
            resolution: Grid resolution
            obstacles: List of obstacles
            
        Returns:
            risk_map: 2D numpy array of risk values
            X, Y: Meshgrid for plotting
        """
        x = np.linspace(x_range[0], x_range[1], int((x_range[1]-x_range[0])/resolution))
        y = np.linspace(y_range[0], y_range[1], int((y_range[1]-y_range[0])/resolution))
        X, Y = np.meshgrid(x, y)
        
        risk_map = np.zeros_like(X)
        
        # Calculate risk at each point using the distributionally robust approach
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                position = np.array([X[i, j], Y[i, j]])
                risk_map[i, j] = self.calculate_total_risk(position, obstacles)
                
        return risk_map, X, Y
import numpy as np
import cvxpy as cvx
from scipy.stats import chi2

class CVaRObstacleAvoidance:
    def __init__(self, confidence_level=0.95, safety_margin=0.1):
        self.confidence_level = confidence_level
        # Scale safety margin based on confidence level
        self.base_safety_margin = safety_margin
        self.safety_margin = self._get_scaled_safety_margin()
        self.chi2_val = chi2.ppf(confidence_level, df=3)  # Changed to df=3 for full 3D
    
    def _get_scaled_safety_margin(self):
        """Scale safety margin based on confidence level"""
        # Ensure a reasonable minimum safety margin even at low confidence
        base_confidence = 0.85  # Lower base confidence
        scaling_factor = 2.0    # Gentler scaling
        
        # Apply a non-linear scaling that increases more rapidly at higher confidence levels
        confidence_effect = max(0, (self.confidence_level - base_confidence) / (1.0 - base_confidence))
        # Add a minimum safety factor (0.8) to ensure even low confidence has some safety
        return self.base_safety_margin * (0.8 + scaling_factor * confidence_effect)
    
    def calculate_cvar_constraint(self, pos_var, obstacle, t, min_dist):
        """Calculate collision avoidance constraints using distributionally robust approach."""
        constraints = []
        params = obstacle.get_constraint_parameters()
        center = params['center']
        
        # Handle dimension differences consistently
        pos_dim = pos_var.shape[0]
        center_dim = center.shape[0]
        
        if pos_dim != center_dim:
            if pos_dim > center_dim:
                # If position has higher dimension, truncate to match obstacle
                pos_var_adj = pos_var[:center_dim]
            else:
                # If obstacle has higher dimension, truncate to match position
                center_adj = center[:pos_dim]
                center = center_adj
        else:
            pos_var_adj = pos_var
        
        # Scale minimum distance based on confidence level
        min_safety_factor = 0.8  # Minimum safety factor even at low confidence
        confidence_scale = (self.confidence_level - 0.85) / (1.0 - 0.85) 
        confidence_scale = max(0, confidence_scale)  # Ensure non-negative
        scaled_min_dist = min_dist * (min_safety_factor + confidence_scale)
        
        # Base constraint for current obstacle position
        constraints.append(
            cvx.sum_squares(pos_var_adj - center) >= scaled_min_dist**2
        )
        
        # Early return if no uncertainty data
        uncertainty_idx = min(t-1, len(params['uncertainty'])-1)
        if uncertainty_idx < 0 or 'uncertainty' not in params:
            return constraints
        
        # Get uncertainty for current prediction time
        uncertainty = params['uncertainty'][uncertainty_idx]
        
        # Skip if no uncertainty data available
        if not uncertainty or 'means' not in uncertainty:
            return constraints
            
        means = uncertainty['means']
        covariances = uncertainty['covariances']
        weights = uncertainty['weights']
        
        # Apply constraints for each component in the ambiguity set
        for j, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
            if weight < 0.05:  # Skip negligible components
                continue
            
            # Match dimensionality between mean and position variable
            if len(mean) != pos_dim:
                if len(mean) > pos_dim:
                    mean_adj = mean[:pos_dim]
                    mean = mean_adj
                    
                    # Also adjust covariance matrix
                    cov_adj = cov[:pos_dim, :pos_dim]
                    cov = cov_adj
                else:
                    # Skip if dimensions can't be reconciled
                    continue
                
            # Regularize covariance for numerical stability
            reg_cov = cov + np.eye(len(cov)) * 1e-6
            
            try:
                # Current position of obstacle
                curr_pos = np.array(center)[:pos_dim]  # Ensure matching dimensions
                
                # Vector from predicted mean to position
                vec_to_robot = curr_pos - mean
                
                # Handle special case when vector is very small
                if np.linalg.norm(vec_to_robot) < 1e-6:
                    # Use squared distance constraint
                    # Scale with confidence level
                    weighted_min_dist = scaled_min_dist * (1.0 + weight)
                    constraints.append(
                        cvx.sum_squares(pos_var_adj - mean) >= weighted_min_dist**2
                    )
                    continue
                
                # Normalize to get direction
                direction = vec_to_robot / np.linalg.norm(vec_to_robot)
                
                # Calculate scale factor based on chi-square distribution
                # Now the scale increases with confidence (chi2_val increases)
                variance = direction.T @ reg_cov @ direction
                scale_factor = np.sqrt(self.chi2_val * variance)
                
                # Adjust minimum distance based on component weight and confidence
                weighted_min_dist = scaled_min_dist * (1.0 + weight)
                
                # Add linearized constraint (DCP-compliant)
                # More conservative constraint for higher confidence levels
                constraints.append(
                    direction @ (pos_var_adj - mean) >= weighted_min_dist + scale_factor
                )
                
            except Exception as e:
                print(f"Constraint generation failed: {e}")
                # Use fallback constraint
                weighted_min_dist = scaled_min_dist * (1.0 + weight)
                constraints.append(
                    cvx.sum_squares(pos_var_adj - mean) >= weighted_min_dist**2
                )
        
        return constraints
    
    def calculate_risk(self, position, obstacle):
        """Calculate distributionally robust risk of collision with an obstacle."""
        params = obstacle.get_constraint_parameters()
        center = params['center']
        
        # Handle dimensionality differences properly
        pos_dim = len(position)
        center_dim = len(center)
        
        if pos_dim != center_dim:
            if pos_dim > center_dim:
                # If position is higher dimension (e.g., 3D) but obstacle is lower (e.g., 2D)
                # Use only matching coordinates
                position_adj = position[:center_dim]
                dist = np.linalg.norm(position_adj - center)
            else:
                # If position is lower dimension but obstacle is higher
                # Use only matching coordinates of obstacle
                center_adj = center[:pos_dim]
                dist = np.linalg.norm(position - center_adj)
        else:
            # Same dimensionality, direct comparison
            dist = np.linalg.norm(position - center)
            
        radius = params['radius']
        # Scale safety margin with confidence level
        min_safe_dist = radius + self.safety_margin
        
        if dist <= min_safe_dist:
            risk = 1.0
        else:
            # Make risk decrease slower with higher confidence levels
            # Risk decreases with distance according to paper's model
            min_confidence_effect = 0.8  # Minimum effect at lowest confidence
            confidence_effect = min_confidence_effect + (self.confidence_level - 0.85) * 2.0
            risk = (min_safe_dist / dist) ** (2.0 / confidence_effect)
            
        # Add risk from ambiguity set components
        if 'uncertainty' in params and len(params['uncertainty']) > 0:
            uncertainty = params['uncertainty'][0]  # First time step prediction
            
            # Handle case where uncertainty data is incomplete
            if not uncertainty or 'means' not in uncertainty:
                return risk
                
            # Consider each mixture component
            for mean, cov, weight in zip(
                uncertainty.get('means', []), 
                uncertainty.get('covariances', []),
                uncertainty.get('weights', [])
            ):
                if weight < 0.05:  # Skip negligible components
                    continue
                
                # Handle dimensionality differences for mean
                mean_dim = len(mean)
                if pos_dim != mean_dim:
                    if pos_dim > mean_dim:
                        # Use only the matching coordinates of position
                        position_adj = position[:mean_dim]
                        comp_dist = np.linalg.norm(position_adj - mean)
                    else:
                        # Use only the matching coordinates of mean
                        mean_adj = mean[:pos_dim]
                        comp_dist = np.linalg.norm(position - mean_adj)
                else:
                    # Same dimensionality, direct comparison
                    comp_dist = np.linalg.norm(position - mean)
                
                try:
                    # Match cov dimensions to handle properly
                    if len(cov) > pos_dim:
                        cov_adj = cov[:pos_dim, :pos_dim]
                    elif len(cov) < pos_dim:
                        # Skip if cov is too small
                        continue
                    else:
                        cov_adj = cov
                        
                    # Calculate risk based on uncertainty model
                    eigvals = np.linalg.eigvalsh(cov_adj)
                    max_std = np.sqrt(max(eigvals))
                    
                    # Exponential risk model - now higher confidence means higher risk
                    # Instead of dividing by chi2_val, we multiply to increase risk with confidence
                    confidence_scale = max(1.0, self.chi2_val / 5.0)  # Normalize to reasonable range
                    comp_risk = weight * np.exp(-(comp_dist**2) / (2 * max_std**2 * confidence_scale))
                    
                    # Combine risks with proper weighting
                    risk = risk + comp_risk * (1.0 - risk)
                    
                except (np.linalg.LinAlgError, ValueError) as e:
                    # Fallback risk calculation
                    # Make more conservative with higher confidence
                    confidence_factor = 1.0 + (self.confidence_level - 0.9) * 3.0
                    comp_risk = weight * (min_safe_dist / max(comp_dist, min_safe_dist)) ** (2.0 / confidence_factor)
                    risk = max(risk, comp_risk)
        
        return min(risk, 1.0)  # Ensure risk is bounded
    
    def calculate_total_risk(self, position, obstacles):
        """Calculate total risk from all obstacles using distributionally robust approach."""
        if not obstacles:
            return 0.0
            
        total_risk = 0.0
        
        # Calculate risk from each obstacle
        for obs in obstacles:
            obs_risk = self.calculate_risk(position, obs)
            
            # Combine risks using method described in paper
            total_risk = total_risk + obs_risk * (1.0 - total_risk)
            
        return total_risk
    
    def generate_risk_map(self, x_range, y_range, resolution, obstacles):
        """Generate a 2D risk map for visualization."""
        x = np.linspace(x_range[0], x_range[1], int((x_range[1]-x_range[0])/resolution))
        y = np.linspace(y_range[0], y_range[1], int((y_range[1]-y_range[0])/resolution))
        X, Y = np.meshgrid(x, y)
        
        risk_map = np.zeros_like(X)
        
        # Calculate risk at each point using the distributionally robust approach
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Use average z-height from obstacles for risk calculation in 3D
                z_height = 0.0
                count = 0
                for obs in obstacles:
                    if len(obs.get_position()) > 2:  # Check if obstacle is 3D
                        z_height += obs.get_position()[2]
                        count += 1
                
                if count > 0:
                    z_height /= count
                    # Create a 3D position with estimated z-height
                    position = np.array([X[i, j], Y[i, j], z_height])
                else:
                    position = np.array([X[i, j], Y[i, j]])
                    
                risk_map[i, j] = self.calculate_total_risk(position, obstacles)
                
        return risk_map, X, Y
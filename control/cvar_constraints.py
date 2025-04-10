import numpy as np
import cvxpy as cvx
from scipy.stats import chi2

class CVaRObstacleAvoidance:
    def __init__(self, confidence_level=0.95, safety_margin=0.1):
        self.confidence_level = confidence_level
        self.safety_margin = safety_margin
        self.chi2_val = chi2.ppf(confidence_level, df=2)
    
    def calculate_cvar_constraint(self, pos_var, obstacle, t, min_dist):
        constraints = []
        params = obstacle.get_constraint_parameters()
        center = params['center']
        
        uncertainty_idx = min(t-1, len(params['uncertainty'])-1)
        
        if uncertainty_idx < 0:
            dist_var = cvx.norm(pos_var - center)
            constraints.append(dist_var >= min_dist)
            return constraints
        
        uncertainty = params['uncertainty'][uncertainty_idx]
        means = uncertainty['means']
        covariances = uncertainty['covariances']
        weights = uncertainty['weights']
        
        for i, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
            if weight < 0.05:
                continue
                
            reg_cov = cov + np.eye(2) * 1e-6
            
            try:
                cov_inv = np.linalg.inv(reg_cov)
                
                curr_pos = np.array(center)
                vec_to_robot = curr_pos - mean
                
                if np.linalg.norm(vec_to_robot) < 1e-6:
                    vec_to_robot = np.array([1.0, 0.0])
                
                direction = vec_to_robot / np.linalg.norm(vec_to_robot)
                
                scale_factor = np.sqrt(self.chi2_val * direction.T @ reg_cov @ direction)
                weighted_min_dist = min_dist * (1.0 + weight)
                
                constraints.append(direction @ (pos_var - mean) >= weighted_min_dist + scale_factor)
                
            except np.linalg.LinAlgError:
                constraints.append(cvx.norm(pos_var - mean) >= min_dist * (1.0 + weight))
                
        return constraints
    
    def calculate_risk(self, position, obstacle):
        params = obstacle.get_constraint_parameters()
        center = params['center']
        risk = 0.0
        
        dist = np.linalg.norm(position - center)
        radius = params['radius']
        min_safe_dist = radius + self.safety_margin
        
        if dist <= min_safe_dist:
            risk = 1.0
        else:
            risk = (min_safe_dist / dist) ** 2
            
        if 'uncertainty' in params and len(params['uncertainty']) > 0:
            uncertainty = params['uncertainty'][0]
            
            for mean, cov, weight in zip(uncertainty['means'], 
                                         uncertainty['covariances'],
                                         uncertainty['weights']):
                if weight < 0.05:
                    continue
                
                comp_dist = np.linalg.norm(position - mean)
                
                try:
                    eigvals = np.linalg.eigvalsh(cov)
                    max_std = np.sqrt(max(eigvals))
                    
                    prob_risk = weight * np.exp(-(comp_dist**2) / (2 * max_std**2 * self.chi2_val))
                    
                    risk = risk + prob_risk * (1.0 - risk)
                    
                except np.linalg.LinAlgError:
                    comp_risk = weight * (min_safe_dist / max(comp_dist, min_safe_dist)) ** 2
                    risk = max(risk, comp_risk)
        
        return risk
    
    def calculate_total_risk(self, position, obstacles):
        total_risk = 0.0
        
        for obs in obstacles:
            obs_risk = self.calculate_risk(position, obs)
            total_risk = total_risk + obs_risk * (1.0 - total_risk)
            
        return total_risk
    
    def generate_risk_map(self, x_range, y_range, resolution, obstacles):
        x = np.linspace(x_range[0], x_range[1], int((x_range[1]-x_range[0])/resolution))
        y = np.linspace(y_range[0], y_range[1], int((y_range[1]-y_range[0])/resolution))
        X, Y = np.meshgrid(x, y)
        
        risk_map = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                position = np.array([X[i, j], Y[i, j]])
                risk_map[i, j] = self.calculate_total_risk(position, obstacles)
                
        return risk_map, X, Y
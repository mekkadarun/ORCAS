import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional, Union


def calculate_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate the Conditional Value-at-Risk (CVaR) for a set of loss values.
    
    Args:
        losses: Array of loss values
        alpha: Confidence level (typically 0.95 or 0.99)
        
    Returns:
        CVaR value
    """
    if len(losses) == 0:
        return 0.0
        
    # Sort losses
    sorted_losses = np.sort(losses)
    
    # Calculate the Value-at-Risk (VaR) at alpha level
    var_index = int(np.ceil((1 - alpha) * len(sorted_losses))) - 1
    var_index = max(0, var_index)  # Ensure non-negative index
    var = sorted_losses[var_index]
    
    # Calculate CVaR as mean of losses above VaR
    tail_losses = sorted_losses[var_index:]
    cvar = np.mean(tail_losses)
    
    return cvar


def formulate_cvar_constraint(
    robot_position: cp.Variable,
    obstacle_position: np.ndarray,
    components: List[Dict],
    safety_radius_sq: float,
    alpha: float = 0.95
) -> Tuple[cp.Expression, cp.Variable]:
    """
    Formulate the CVaR constraint for distributionally robust optimization.
    
    Args:
        robot_position: CVXPY variable for robot position
        obstacle_position: Position of the obstacle
        components: GMM components (weights, means, covariances)
        safety_radius_sq: Squared safety radius
        alpha: Confidence level
        
    Returns:
        Tuple of (CVaR expression, auxiliary variable)
    """
    # Auxiliary variable for CVaR calculation
    z = cp.Variable(1)
    
    # Initialize CVaR expression
    cvar_expr = 0
    
    # For each GMM component
    for component in components:
        weight = component['weight']
        mean = component['mean']
        cov = component['covariance']
        
        # Calculate expected squared distance
        # E[||robot_pos - (obstacle_pos + movement)||^2]
        expected_dist_sq = cp.sum_squares(robot_position - obstacle_position) - \
                           2 * robot_position.T @ mean + \
                           obstacle_position.T @ mean + \
                           cp.sum(np.diag(cov)) + \
                           mean.T @ mean
        
        # CVaR contribution for this component
        component_cvar = z + (1/(1-alpha)) * cp.maximum(0, expected_dist_sq - safety_radius_sq - z)
        
        # Add weighted contribution to total CVaR
        cvar_expr += weight * component_cvar
    
    return cvar_expr, z


def dist_robust_cvar_constraint(
    robot_position: cp.Variable,
    obstacle_position: np.ndarray,
    ambiguity_components: List[Dict],
    safety_radius_sq: float,
    alpha: float = 0.95
) -> List[cp.Constraint]:
    """
    Create the distributionally robust CVaR constraint.
    
    Args:
        robot_position: CVXPY variable for robot position
        obstacle_position: Position of the obstacle
        ambiguity_components: Components of the ambiguity set
        safety_radius_sq: Squared safety radius
        alpha: Confidence level
        
    Returns:
        List of CVXPY constraints
    """
    cvar_expr, z = formulate_cvar_constraint(
        robot_position, obstacle_position, ambiguity_components, safety_radius_sq, alpha
    )
    
    # The constraint is that CVaR >= 0
    # Which means risk of unsafe distance is limited to alpha
    return [cvar_expr >= 0]

def gaussian_cvar_approximation(mean: float, std_dev: float, alpha: float = 0.95) -> float:
    """
    Analytical approximation of CVaR for a Gaussian distribution.
    
    Args:
        mean: Mean of the distribution
        std_dev: Standard deviation
        alpha: Confidence level
    
    Returns:
        Approximate CVaR value
    """
    from scipy.stats import norm
    
    # Handle numerical edge cases
    if std_dev < 1e-10:
        return mean  # If variance is essentially zero, return mean
        
    try:
        # Prevent numerical issues
        alpha = min(max(alpha, 0.001), 0.999)
        var = mean - norm.ppf(alpha) * std_dev  # Value-at-Risk
        cvar = mean - std_dev * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
        
        # Handle potential numerical issues
        if not np.isfinite(cvar):
            return mean
            
        return cvar
    except:
        # Fall back to a conservative approximation
        return mean - 3 * std_dev  # Roughly equivalent to alpha=0.99

def simplified_cvar_constraint(
    robot_position: cp.Variable,
    obstacle_position: np.ndarray,
    components: List[Dict],
    safety_radius: float,
    alpha: float = 0.95
) -> List[cp.Constraint]:
    """
    Create simplified CVaR constraints using analytical approximation.
    
    Args:
        robot_position: CVXPY variable for robot position
        obstacle_position: Position of the obstacle
        components: GMM components (weights, means, covariances)
        safety_radius: Safety radius
        alpha: Confidence level
        
    Returns:
        List of CVXPY constraints
    """
    constraints = []
    
    # For each GMM component
    for component in components:
        weight = component['weight']
        mean = component['mean']
        cov = component['covariance']
        
        # Project the Gaussian distribution along multiple directions
        # This is a simplification, but should work well for convex obstacles
        directions = [
            np.array([1.0, 0.0, 0.0]),  # x-axis
            np.array([0.0, 1.0, 0.0]),  # y-axis
            np.array([0.0, 0.0, 1.0]),  # z-axis
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2),  # xy diagonal
            np.array([1.0, 0.0, 1.0]) / np.sqrt(2),  # xz diagonal
            np.array([0.0, 1.0, 1.0]) / np.sqrt(2)   # yz diagonal
        ]
        
        # For each direction, create a constraint
        for dir_vec in directions:
            # Project the Gaussian distribution along the direction vector
            proj_mean = dir_vec @ mean
            proj_var = dir_vec @ cov @ dir_vec
            proj_std = np.sqrt(proj_var)
            
            # Calculate the minimum distance using CVaR approximation
            min_dist = gaussian_cvar_approximation(proj_mean, proj_std, alpha)
            
            # Create a linear constraint: (x-x_obs)·dir >= d
            # This is a convex constraint and DCP-compliant
            distance = (safety_radius - min_dist) * 0.2
            if distance > 0:  # Only add constraint if the distance is positive
                constraints.append(dir_vec @ (robot_position - obstacle_position) >= distance)
    
    return constraints
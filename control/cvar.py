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
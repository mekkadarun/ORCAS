# ORCAS: Online-Learning-Based Distributionally Robust Motion Control

This repository implements an advanced motion control system for autonomous quadrotor navigation through uncertain environments. The implementation is based on the paper "Online-Learning-Based Distributionally Robust Motion Control".

## Overview

The ORCAS (Online Robust Control for Autonomous Systems) framework provides:

- Real-time obstacle avoidance with uncertainty handling
- Online learning of obstacle movement patterns
- Distributionally robust control using CVaR (Conditional Value-at-Risk)
- Support for both 2D and 3D environments

## Key Features

- **Robust Motion Planning**: Uses model predictive control (MPC) with probabilistic safety guarantees
- **Online Learning**: Continuously updates movement models for dynamic obstacles
- **Gaussian Mixture Models**: Represents complex, multi-modal uncertainty distributions
- **Ambiguity Sets**: Maintains sets of possible distributions for robust decision making
- **CVaR-Based Risk Assessment**: Quantifies and minimizes tail risks for enhanced safety

## System Components

### Motion Models
- `quadrotor.py`: Simple 2D quadrotor dynamics
- `real_quadrotor.py`: Full 3D quadrotor model with realistic dynamics
- `static_obstacle.py`: Fixed obstacles with defined radius
- `dynamic_obstacle.py`: Moving obstacles with constant velocity
- `gmm_obstacle.py`: Obstacles with GMM-based motion prediction
- `real_obstacle.py`: 3D obstacles with advanced uncertainty modeling

### Control Systems
- `static_mpc.py`: Basic MPC for navigation around static obstacles
- `dynamic_mpc.py`: MPC adapted for dynamic obstacles with predicted trajectories
- `gmm_mpc.py`: MPC with CVaR constraints for GMM-based uncertainty
- `real_mpc.py`: Full 3D MPC with enhanced safety guarantees

### Uncertainty Handling
- `ambiguity_set.py`: Implementation of distributional ambiguity sets
- `real_ambiguity_set.py`: Extended 3D ambiguity sets
- `cvar_constraints.py`: CVaR-based collision avoidance constraints
- `real_cvar_constraints.py`: Enhanced 3D collision avoidance with confidence levels

### Visualization and Evaluation
- `static_orcas.py`: Demo for static obstacle navigation
- `dynamic_orcas.py`: Demo for dynamic obstacle avoidance
- `gmm_orcas.py`: Demo with uncertainty visualization
- `real_orcas.py`: Full 3D visualization with orientation display
- `eval.py`: Evaluation framework to assess performance across different confidence levels
- `env.py`: Environment plotting utilities

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- cvxpy (for optimization)

### Running Simulations

```bash
# Run static obstacle simulation
python test/static_orcas.py

# Run dynamic obstacle simulation
python test/dynamic_orcas.py

# Run GMM-based uncertainty simulation with visualization
python test/gmm_orcas.py

# Run full 3D simulation
python test/real_orcas.py
```

### Evaluation

The framework includes tools to evaluate performance across different confidence levels:

```bash
# Run evaluation with 10 runs per confidence level
python sim/eval.py --runs 10 --seed 42 --output results/confidence_eval
```

## Method

The approach combines three key techniques:

1. **Online Learning**: The system continuously observes and learns the movement patterns of obstacles using Gaussian Mixture Models.

2. **Distributional Robustness**: Instead of assuming a single probability distribution, the controller considers a set of possible distributions (an ambiguity set) to provide robustness against uncertainty.

3. **CVaR-based Risk Assessment**: The controller assesses risk using Conditional Value-at-Risk, which focuses on the worst-case scenarios rather than just the average case.

## Applications

- Autonomous drone navigation in dynamic environments
- Robot motion planning with uncertainty
- Safe navigation in crowds
- Collision avoidance systems with probabilistic guarantees

## License

This project is licensed under the MIT License - see the LICENSE file for details.

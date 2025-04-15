import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
from tqdm import tqdm
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import transforms

from motion_models.real_quadrotor import RealQuadrotor
from motion_models.real_obstacle import GMMObstacle
from control.real_mpc import RealCVaRGMMMPC_3D

class ConfidenceLevelEvaluator:
    """
    Evaluates how different confidence levels affect quadrotor performance
    in the distributionally robust motion control implementation.
    """
    def __init__(self, seed=42):
        # Default goal position
        self.goal = np.array([7.0, 7.0, 3.0])
        
        # Results storage
        self.results = []
        
        # Set fixed random seed for reproducibility
        self.seed = seed
        np.random.seed(seed)
        
        # Store predefined movement patterns for consistency
        self.movement_patterns = [
            ([0.1, 0.2, 0.0], [[0.02, 0.005, 0.001], [0.005, 0.02, 0.001], [0.001, 0.001, 0.005]]),
            ([0.1, -0.1, 0.0], [[0.03, -0.01, 0.0005], [-0.01, 0.03, 0.0005], [0.0005, 0.0005, 0.003]]),
            ([-0.1, 0.15, 0.0], [[0.04, 0.02, 0.0008], [0.02, 0.04, 0.0008], [0.0008, 0.0008, 0.004]])
        ]
        
        # Standard obstacle positions
        self.standard_positions = [
            (2.0, 2.0, 0.0),  # Z will be set to goal_z
            (4.0, 4.0, 0.0),
            (2.70, 1.0, 0.0),
            (5.0, 5.0, 0.0),
            (6.0, 6.0, 0.0)
        ]
        
    def setup_scenario(self, confidence_level=0.95, obstacle_count=5, obstacle_radius=0.3, scenario_seed=None):
        """
        Set up a standard test scenario with specified parameters.
        
        Args:
            confidence_level: Confidence level for CVaR computation
            obstacle_count: Number of obstacles in the environment
            obstacle_radius: Radius of the obstacles
            scenario_seed: Seed for this specific scenario (combines with base seed)
        """
        # Set scenario-specific seed for reproducibility across confidence levels
        if scenario_seed is not None:
            np.random.seed(self.seed + scenario_seed)
        
        # Initial state: [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]
        initial_state = np.zeros(12)
        initial_state[:3] = np.array([0.0, 0.0, 0.0])  # Initial position
        
        self.quadrotor = RealQuadrotor(initial_state)
        
        # Goal height to use for obstacles
        goal_z = self.goal[2]
        
        # Create obstacles at fixed positions for consistency
        self.obstacles = []
        for i in range(min(obstacle_count, len(self.standard_positions))):
            x, y, _ = self.standard_positions[i]
            self.obstacles.append(
                GMMObstacle((x, y, goal_z), radius=obstacle_radius, n_components=2, is_3d=True)
            )
                
        # Initialize controller with specified confidence level
        self.mpc = RealCVaRGMMMPC_3D(
            horizon=10, 
            dt=0.1, 
            quad_radius=0.3, 
            confidence_level=confidence_level
        )
        
        # Pre-train GMM models with consistent patterns
        self._pretrain_obstacle_models(scenario_seed)
    
    def _pretrain_obstacle_models(self, scenario_seed=None):
        """
        Pre-train GMM models with consistent movement patterns.
        Uses deterministic initialization for fair comparison.
        """
        # Use scenario seed if provided
        if scenario_seed is not None:
            np.random.seed(self.seed + scenario_seed + 100)  # Different offset for training
        
        # Generate initial movement data for each obstacle
        for i, obs in enumerate(self.obstacles):
            pattern_idx = i % len(self.movement_patterns)
            mean, cov = self.movement_patterns[pattern_idx]
            
            # Use deterministic samples with small offsets rather than random
            for j in range(30):
                if scenario_seed is not None:
                    # Deterministic sample with minor variations
                    offset = np.sin(j * 0.1 + i * 0.3) * 0.05
                    movement = np.array(mean) + offset
                else:
                    # Standard random sample
                    movement = np.random.multivariate_normal(mean=mean, cov=cov)
                
                obs.movement_history.append(movement)
                obs.ambiguity_set.add_movement_data(movement)
            
            # Add a few controlled outliers
            for j in range(5):
                if scenario_seed is not None:
                    # Deterministic outlier
                    offset = np.sin(j * 0.5 + i * 0.7) * 0.15
                    outlier = -np.array(mean) + offset
                else:
                    # Random outlier
                    outlier = np.random.multivariate_normal(
                        mean=[-m for m in mean],
                        cov=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
                    )
                
                obs.movement_history.append(outlier)
                obs.ambiguity_set.add_movement_data(outlier)
            
            # Initialize GMM and ambiguity set
            obs._fit_gmm()
            obs.ambiguity_set.update_mixture_model()
        
        # Reset random seed to base seed
        np.random.seed(self.seed)
    
    def run_simulation(self, max_steps=300, scenario_name=None, use_fixed_sequence=False, fixed_sequence=None):
        """
        Run a simulation with the current configuration.
        
        Args:
            max_steps: Maximum number of simulation steps
            scenario_name: Name of the scenario for tracking
            use_fixed_sequence: Whether to use a fixed random number sequence
            fixed_sequence: Pre-generated random sequence to use
            
        Returns:
            Dictionary with simulation results and random sequence used
        """
        start_time = time.time()
        
        # Initialize storage
        step_count = 0
        total_cost = 0.0
        actual_trajectory = []
        obstacle_trajectories = [[] for _ in self.obstacles]
        distance_to_goal_history = []
        risk_history = []
        
        # For reproducibility
        if use_fixed_sequence and fixed_sequence is not None:
            random_sequence = fixed_sequence
        else:
            # Generate a random sequence for this run
            random_sequence = np.random.random(max_steps * 10)  # More than we need
        
        sequence_index = 0
        
        success = False
        collision = False
        
        # Record initial obstacle positions
        for i, obs in enumerate(self.obstacles):
            obstacle_trajectories[i].append(obs.get_position().copy())
        
        # Run simulation
        for step in range(max_steps):
            current_state = self.quadrotor.get_state()
            position = current_state[:3]
            actual_trajectory.append(position.copy())
            
            # Calculate distance to goal
            distance_to_goal = np.linalg.norm(position - self.goal)
            distance_to_goal_history.append(distance_to_goal)
            
            # Calculate current risk
            current_risk = self.mpc.cvar_avoidance.calculate_total_risk(position, self.obstacles)
            risk_history.append(current_risk)
            
            # Check if goal is reached - use consistent criteria
            if distance_to_goal < 0.5:
                success = True
                break
            
            # Update obstacle positions - use standard update
            for i, obs in enumerate(self.obstacles):
                # Control randomness using the seed, but don't try to override
                # the movement directly since that method parameter doesn't exist
                obs.update_position(self.mpc.dt, use_gmm=True, maintain_z=True, goal_z=self.goal[2])
                
                obstacle_trajectories[i].append(obs.get_position().copy())
            
            # Generate control sequence using MPC
            control_sequence = self.mpc.optimize_trajectory(current_state, self.goal, self.obstacles)
            
            if control_sequence is not None:
                control_cost = np.sum(np.linalg.norm(control_sequence[0])**2)
                
                self.quadrotor.update_state(control_sequence[0], self.mpc.dt)
                
                new_state = self.quadrotor.get_state()
                state_cost = np.linalg.norm(new_state[:3] - self.goal)**2
                
                # More reasonable cost calculation with capping
                step_cost = state_cost + 0.1 * control_cost
                step_cost = min(step_cost, 100)  # Cap to prevent huge costs
                total_cost += step_cost
            else:
                print(f"Warning: No valid control found in step {step}")
                # Apply zero control as fallback
                self.quadrotor.update_state(np.zeros(4), self.mpc.dt)
                
            step_count += 1
            
            # Check for collisions - consistent check
            for obs in self.obstacles:
                dist = np.linalg.norm(position - obs.get_position())
                if dist <= (self.quadrotor.radius + obs.radius):
                    collision = True
                    break
                    
            if collision:
                success = False
                break
        
        # Calculate metrics with more detailed data
        completion_time = time.time() - start_time
        final_position = self.quadrotor.get_state()[:3]
        goal_error = np.linalg.norm(final_position - self.goal)
        
        # Calculate path efficiency (ratio of straight-line distance to actual path length)
        straight_line_distance = np.linalg.norm(self.goal - actual_trajectory[0])
        actual_path_length = 0.0
        for i in range(1, len(actual_trajectory)):
            actual_path_length += np.linalg.norm(np.array(actual_trajectory[i]) - np.array(actual_trajectory[i-1]))
        
        path_efficiency = straight_line_distance / max(actual_path_length, 0.001)
        
        # Calculate average risk along trajectory
        average_risk = sum(risk_history) / len(risk_history) if risk_history else 0
        
        # Calculate average minimum distance to obstacles with more detailed tracking
        total_min_dist = 0.0
        count = 0
        min_distances = []
        
        for i, pos in enumerate(actual_trajectory):
            min_dist = float('inf')
            for j, obs_traj in enumerate(obstacle_trajectories):
                if i < len(obs_traj):
                    dist = np.linalg.norm(pos - obs_traj[i]) - self.obstacles[j].radius
                    min_dist = min(min_dist, dist)
            
            if min_dist != float('inf'):
                total_min_dist += min_dist
                count += 1
                min_distances.append(min_dist)
                
        avg_min_distance = total_min_dist / count if count > 0 else 0
        
        # Calculate minimum separation distance throughout trajectory
        min_separation = min(min_distances) if min_distances else 0
        
        # Return detailed results
        return {
            'success': success,
            'collision': collision,
            'steps': step_count,
            'total_cost': total_cost,
            'goal_error': goal_error,
            'path_efficiency': path_efficiency,
            'avg_min_distance': avg_min_distance,
            'min_separation': min_separation,
            'average_risk': average_risk,
            'computation_time': completion_time,
            'trajectory': actual_trajectory,
            'distance_history': distance_to_goal_history,
            'risk_history': risk_history,
            'scenario': scenario_name,
            'random_sequence': random_sequence if use_fixed_sequence else None
        }
    
    def evaluate_confidence_levels(self, confidence_levels, runs_per_level=10, use_identical_scenarios=True):
        """
        Evaluate performance across different confidence levels with improved statistical rigor.
        
        Args:
            confidence_levels: List of confidence levels to test
            runs_per_level: Number of runs per confidence level (increased for better statistics)
            use_identical_scenarios: Whether to use identical scenarios across confidence levels
            
        Returns:
            DataFrame with results
        """
        all_results = []
        
        # Generate fixed random seeds for each run
        scenario_seeds = []
        for i in range(runs_per_level):
            scenario_seeds.append(i + 1)
        
        # Run evaluations for each confidence level
        for level_idx, level in enumerate(tqdm(confidence_levels, desc="Evaluating confidence levels")):
            for run_idx in range(runs_per_level):
                # Use consistent scenario seeds across confidence levels
                if use_identical_scenarios:
                    scenario_seed = scenario_seeds[run_idx]
                else:
                    scenario_seed = level_idx * 100 + run_idx + 1
                
                # Setup scenario with deterministic seed
                self.setup_scenario(
                    confidence_level=level,
                    scenario_seed=scenario_seed
                )
                
                # Run simulation
                scenario_name = f"Confidence {level*100:.0f}% (Run {run_idx+1})"
                result = self.run_simulation(
                    scenario_name=scenario_name,
                    use_fixed_sequence=False  # Can't use fixed sequence with current implementation
                )
                
                # Add metadata to result
                result['confidence_level'] = level
                result['run'] = run_idx
                result['scenario_seed'] = scenario_seed
                
                all_results.append(result)
                
                # Reset random seed after each run
                np.random.seed(self.seed)
                
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Calculate statistical measures if enough data
        if len(results_df) >= len(confidence_levels) * 3:  # Need at least 3 runs per level
            self._add_statistical_analysis(results_df)
        
        return results_df
    
    def _add_statistical_analysis(self, results_df):
        """Add statistical significance tests to results"""
        try:
            from scipy import stats

            # Group by confidence level
            grouped = results_df.groupby('confidence_level')
            
            # Store baseline for comparison (95% confidence level)
            baseline_idx = results_df['confidence_level'] == 0.95
            if baseline_idx.sum() > 0:
                baseline_efficiency = results_df.loc[baseline_idx, 'path_efficiency']
                baseline_steps = results_df.loc[baseline_idx, 'steps']
                baseline_risk = results_df.loc[baseline_idx, 'average_risk']
                
                # Add p-values for each confidence level compared to baseline
                p_values = []
                
                for level, group in grouped:
                    if level == 0.95:
                        # Skip baseline
                        p_values.append({
                            'confidence_level': level,
                            'p_value_efficiency': 1.0,
                            'p_value_steps': 1.0,
                            'p_value_risk': 1.0,
                            'significant_diff': False
                        })
                        continue
                    
                    # T-test for efficiency
                    t_stat, p_val_eff = stats.ttest_ind(
                        group['path_efficiency'], 
                        baseline_efficiency,
                        equal_var=False
                    )
                    
                    # T-test for steps
                    t_stat, p_val_steps = stats.ttest_ind(
                        group['steps'], 
                        baseline_steps,
                        equal_var=False
                    )
                    
                    # T-test for risk
                    t_stat, p_val_risk = stats.ttest_ind(
                        group['average_risk'], 
                        baseline_risk,
                        equal_var=False
                    )
                    
                    # Check if difference is significant (p < 0.05)
                    is_significant = (p_val_eff < 0.05 or p_val_steps < 0.05 or p_val_risk < 0.05)
                    
                    p_values.append({
                        'confidence_level': level,
                        'p_value_efficiency': p_val_eff,
                        'p_value_steps': p_val_steps,
                        'p_value_risk': p_val_risk,
                        'significant_diff': is_significant
                    })
                
                # Store p-values in results_df for later use
                self.p_values_df = pd.DataFrame(p_values)
        except ImportError:
            print("SciPy not found. Skipping statistical analysis.")
    
    def visualize_confidence_results(self, results_df, output_file=None):
        """
        Create intuitive visualizations of confidence level effects with statistical significance.
        
        Args:
            results_df: DataFrame with simulation results
            output_file: Optional file path to save visualizations
        """
        # Set up the figure with a grid layout
        plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # Set Seaborn style for better aesthetics
        sns.set_style("whitegrid")
        
        # Color palette for consistent colors
        palette = sns.color_palette("viridis", len(results_df['confidence_level'].unique()))
        
        # 1. Success Rate by Confidence Level (Bar chart with confidence intervals)
        ax1 = plt.subplot(gs[0, 0])
        success_data = results_df.groupby('confidence_level')['success'].agg(['mean', 'count']).reset_index()
        success_data['success_percent'] = success_data['mean'] * 100
        
        # Calculate confidence intervals (95% CI)
        z = 1.96  # 95% confidence
        success_data['ci'] = z * np.sqrt((success_data['mean'] * (1 - success_data['mean'])) / success_data['count'])
        success_data['ci_lower'] = np.maximum(0, success_data['success_percent'] - success_data['ci'] * 100)
        success_data['ci_upper'] = np.minimum(100, success_data['success_percent'] + success_data['ci'] * 100)
        
        # Plot with error bars
        bars = sns.barplot(x='confidence_level', y='success_percent', data=success_data, 
                  palette=palette, ax=ax1)
        
        # Add error bars
        for i, row in success_data.iterrows():
            ax1.errorbar(i, row['success_percent'], 
                       yerr=[[row['success_percent']-row['ci_lower']], [row['ci_upper']-row['success_percent']]], 
                       fmt='none', c='black', capsize=5)
        
        # Add value labels on top of bars
        for i, p in enumerate(bars.patches):
            success_val = success_data.iloc[i]['success_percent']
            ci_text = f"±{success_data.iloc[i]['ci']*100:.1f}%"
            bars.annotate(f"{success_val:.1f}%\n{ci_text}", 
                         (p.get_x() + p.get_width()/2., p.get_height() + 3), 
                         ha='center', va='bottom', fontsize=10, color='black')
        
        ax1.set_xlabel('Confidence Level', fontsize=12)
        ax1.set_ylabel('Success Rate (%)', fontsize=12)
        ax1.set_title('Success Rate by Confidence Level', fontsize=14, fontweight='bold')
        ax1.set_xticklabels([f"{x:.0%}" for x in success_data['confidence_level']])
        ax1.set_ylim(0, 115)  # Leave space for annotations
        
        # 2. Path Efficiency by Confidence Level (Box plot with significance)
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(x='confidence_level', y='path_efficiency', data=results_df, 
                   palette=palette, ax=ax2)
        
        # Add individual points for better visibility
        sns.stripplot(x='confidence_level', y='path_efficiency', data=results_df,
                     color='black', size=4, alpha=0.4, ax=ax2)
        
        ax2.set_xlabel('Confidence Level', fontsize=12)
        ax2.set_ylabel('Path Efficiency', fontsize=12)
        ax2.set_title('Path Efficiency Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticklabels([f"{x:.0%}" for x in sorted(results_df['confidence_level'].unique())])
        
        # Add mean values and significance markers
        means = results_df.groupby('confidence_level')['path_efficiency'].mean()
        
        # Add significance markers if p-values are available
        if hasattr(self, 'p_values_df'):
            for i, level in enumerate(sorted(results_df['confidence_level'].unique())):
                # Skip baseline
                if level == 0.95:
                    ax2.text(i, means[level] + 0.02, f"{means[level]:.3f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                    continue
                
                # Get p-value for this level
                p_val = self.p_values_df[self.p_values_df['confidence_level'] == level]['p_value_efficiency'].values
                if len(p_val) > 0:
                    p_val = p_val[0]
                    # Add significance marker
                    sig_text = ""
                    if p_val < 0.001:
                        sig_text = " ***"
                    elif p_val < 0.01:
                        sig_text = " **"
                    elif p_val < 0.05:
                        sig_text = " *"
                        
                    ax2.text(i, means[level] + 0.02, f"{means[level]:.3f}{sig_text}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    ax2.text(i, means[level] + 0.02, f"{means[level]:.3f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # No p-values, just add means
            for i, level in enumerate(sorted(results_df['confidence_level'].unique())):
                ax2.text(i, means[level] + 0.02, f"{means[level]:.3f}", 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Steps to Goal by Confidence Level (Box plot with significance)
        ax3 = plt.subplot(gs[0, 2])
        sns.boxplot(x='confidence_level', y='steps', data=results_df, 
                   palette=palette, ax=ax3)
        
        # Add individual points for better visibility
        sns.stripplot(x='confidence_level', y='steps', data=results_df,
                     color='black', size=4, alpha=0.4, ax=ax3)
        
        ax3.set_xlabel('Confidence Level', fontsize=12)
        ax3.set_ylabel('Steps to Goal', fontsize=12)
        ax3.set_title('Steps to Goal Distribution', fontsize=14, fontweight='bold')
        ax3.set_xticklabels([f"{x:.0%}" for x in sorted(results_df['confidence_level'].unique())])
        
        # Add mean values and significance markers for steps
        means = results_df.groupby('confidence_level')['steps'].mean()
        
        # Add significance markers if p-values are available
        if hasattr(self, 'p_values_df'):
            for i, level in enumerate(sorted(results_df['confidence_level'].unique())):
                # Skip baseline
                if level == 0.95:
                    ax3.text(i, means[level] + 5, f"{means[level]:.1f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                    continue
                
                # Get p-value for this level
                p_val = self.p_values_df[self.p_values_df['confidence_level'] == level]['p_value_steps'].values
                if len(p_val) > 0:
                    p_val = p_val[0]
                    # Add significance marker
                    sig_text = ""
                    if p_val < 0.001:
                        sig_text = " ***"
                    elif p_val < 0.01:
                        sig_text = " **"
                    elif p_val < 0.05:
                        sig_text = " *"
                        
                    ax3.text(i, means[level] + 5, f"{means[level]:.1f}{sig_text}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    ax3.text(i, means[level] + 5, f"{means[level]:.1f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # No p-values, just add means
            for i, level in enumerate(sorted(results_df['confidence_level'].unique())):
                ax3.text(i, means[level] + 5, f"{means[level]:.1f}", 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Risk vs. Efficiency Scatterplot with regression line
        ax4 = plt.subplot(gs[1, 0:2])
        
        # Add individual points
        for level in sorted(results_df['confidence_level'].unique()):
            level_data = results_df[results_df['confidence_level'] == level]
            ax4.scatter(level_data['average_risk'], level_data['path_efficiency'], 
                       label=f"{level:.0%}", alpha=0.7, s=80)
            
            # Add mean points
            mean_x = level_data['average_risk'].mean()
            mean_y = level_data['path_efficiency'].mean()
            
            # Add annotation for mean
            ax4.annotate(f"{level:.0%}", 
                        (mean_x, mean_y),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=12, fontweight='bold')
        
        # Add best-fit line to show trend
        coeffs = np.polyfit(results_df['average_risk'], results_df['path_efficiency'], 1)
        poly_fn = np.poly1d(coeffs)
        
        x_range = np.linspace(results_df['average_risk'].min(), results_df['average_risk'].max(), 100)
        ax4.plot(x_range, poly_fn(x_range), '--', color='black', alpha=0.7)
        
        ax4.set_xlabel('Average Risk', fontsize=12)
        ax4.set_ylabel('Path Efficiency', fontsize=12)
        ax4.set_title('Risk vs. Path Efficiency Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(title='Confidence Level', title_fontsize=12)
        
        # 5. Summary Statistics Table with statistical significance
        ax5 = plt.subplot(gs[1, 2])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Aggregate key metrics
        summary = results_df.groupby('confidence_level').agg({
            'success': lambda x: f"{x.mean()*100:.1f}%",
            'steps': 'mean',
            'path_efficiency': 'mean',
            'average_risk': 'mean',
            'min_separation': 'mean'
        }).reset_index()
        
        # Add significance indicators if available
        if hasattr(self, 'p_values_df'):
            # Create a new column for path efficiency with significance markers
            summary['path_efficiency_sig'] = summary.apply(
                lambda row: mark_significance(row['path_efficiency'], 
                                            self.p_values_df, 
                                            row['confidence_level'], 
                                            'p_value_efficiency'), 
                axis=1
            )
            
            # Create a new column for steps with significance markers
            summary['steps_sig'] = summary.apply(
                lambda row: mark_significance(row['steps'], 
                                            self.p_values_df, 
                                            row['confidence_level'], 
                                            'p_value_steps'), 
                axis=1
            )
            
            # Add explanatory text for significance
            ax5.text(0.5, -0.15, "* p<0.05, ** p<0.01, *** p<0.001 (compared to 95% confidence)", 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=8, fontstyle='italic')
            
            # Use the columns with significance markers
            table_columns = ['confidence_level', 'success', 'steps_sig', 'path_efficiency_sig', 
                           'average_risk', 'min_separation']
            col_labels = ['Confidence', 'Success', 'Avg Steps', 'Efficiency', 'Avg Risk', 'Min Dist']
        else:
            # Use regular columns without significance markers
            table_columns = ['confidence_level', 'success', 'steps', 'path_efficiency', 
                           'average_risk', 'min_separation']
            col_labels = ['Confidence', 'Success', 'Avg Steps', 'Efficiency', 'Avg Risk', 'Min Dist']
        
        # Format confidence levels as percentages
        summary['confidence_level'] = summary['confidence_level'].apply(lambda x: f"{x:.0%}")
        
        # Round numeric columns
        for col in ['steps', 'path_efficiency', 'average_risk', 'min_separation']:
            if col in summary.columns:
                summary[col] = summary[col].apply(lambda x: f"{x:.3f}")
        
        # Create table
        table = ax5.table(
            cellText=summary[table_columns].values,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Set column widths
        for i in range(len(col_labels)):
            table.auto_set_column_width(i)
        
        ax5.set_title('Performance Summary', fontsize=14, fontweight='bold')
        
        # Add overall title with increased runs info
        plt.suptitle(f'Impact of Confidence Level on Quadrotor Performance\n(Based on {len(results_df)} runs)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add note about reproducibility
        plt.figtext(0.5, 0.01, 
                  f"Evaluation with seed {self.seed}, {len(results_df)//len(summary)} runs per confidence level",
                  ha="center", fontsize=8, style='italic')
        
        plt.tight_layout()
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
    
    def plot_trajectory_comparison(self, results_df, output_file=None):
        """
        Plot trajectories for different confidence levels using median performance runs.
        
        Args:
            results_df: DataFrame with simulation results
            output_file: Optional file path to save visualizations
        """
        # Select successful runs for each confidence level
        successful_runs = results_df[results_df['success'] == True]
        
        # If no successful runs, use all runs
        if len(successful_runs) == 0:
            successful_runs = results_df
        
        # Group by confidence level and select the median run by path efficiency
        representative_runs = []
        for level in sorted(successful_runs['confidence_level'].unique()):
            level_runs = successful_runs[successful_runs['confidence_level'] == level]
            if len(level_runs) > 0:
                # Find the run with median path efficiency
                median_idx = level_runs['path_efficiency'].rank(method='first').astype(int) == (len(level_runs) // 2 + 1)
                if median_idx.sum() > 0:
                    representative_runs.append(level_runs.loc[median_idx].iloc[0])
                else:
                    # Fallback to first run
                    representative_runs.append(level_runs.iloc[0])
        
        # Convert to DataFrame
        representative_df = pd.DataFrame(representative_runs)
        
        if len(representative_df) == 0:
            print("No representative runs found for trajectory comparison")
            return
        
        # Create the visualization
        plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 3, figure=plt.gcf(), width_ratios=[2, 1, 1])
        
        # 3D trajectory plot
        ax1 = plt.subplot(gs[0, 0], projection='3d')
        
        # Color palette for confidence levels
        palette = sns.color_palette("viridis", len(representative_df))
        
        # Plot trajectories
        for i, (_, run) in enumerate(representative_df.iterrows()):
            traj = np.array(run['trajectory'])
            confidence = run['confidence_level']
            
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    label=f"Confidence {confidence:.0%}",
                    linewidth=2, color=palette[i])
            
            # Mark start and end
            ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker='o', s=100, color=palette[i])
            ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker='*', s=200, color=palette[i])
        
        # Plot goal
        ax1.scatter(self.goal[0], self.goal[1], self.goal[2], 
                   marker='*', s=300, color='green', label='Goal')
        
        # Plot obstacles (last run's obstacle positions)
        if len(representative_df) > 0:
            # Setup a scenario to get obstacle positions
            self.setup_scenario(confidence_level=representative_df['confidence_level'].iloc[0])
            
            for i, obs in enumerate(self.obstacles):
                pos = obs.get_position()
                ax1.scatter(pos[0], pos[1], pos[2], marker='o', s=200, 
                           color='red', alpha=0.5)
                
                # Draw sphere to represent obstacle
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                r = obs.radius
                x = pos[0] + r * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + r * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax1.plot_surface(x, y, z, color='red', alpha=0.1)
        
        # Set consistent view
        ax1.view_init(elev=30, azim=45)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.set_title('3D Trajectories by Confidence Level (Median Performance Runs)', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Distance to goal over time
        ax2 = plt.subplot(gs[0, 1])
        
        for i, (_, run) in enumerate(representative_df.iterrows()):
            distance_history = run['distance_history']
            confidence = run['confidence_level']
            
            ax2.plot(range(len(distance_history)), distance_history, 
                    label=f"Confidence {confidence:.0%}",
                    linewidth=2, color=palette[i])
            
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance to Goal')
        ax2.set_title('Distance to Goal Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True)
        ax2.legend()
        
        # Risk profile over time
        ax3 = plt.subplot(gs[0, 2])
        
        for i, (_, run) in enumerate(representative_df.iterrows()):
            risk_history = run['risk_history']
            confidence = run['confidence_level']
            
            ax3.plot(range(len(risk_history)), risk_history, 
                    label=f"Confidence {confidence:.0%}",
                    linewidth=2, color=palette[i])
            
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Risk Value')
        ax3.set_title('Risk Profile Over Time', fontsize=14, fontweight='bold')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Trajectory comparison saved to {output_file}")
        else:
            plt.show()


# Helper function

def mark_significance(value, p_values_df, confidence_level, p_value_col):
    """Add significance markers to values based on p-values"""
    if confidence_level == 0.95:
        return f"{value:.3f}"
    
    p_val_row = p_values_df[p_values_df['confidence_level'] == confidence_level]
    if len(p_val_row) == 0:
        return f"{value:.3f}"
        
    p_val = p_val_row[p_value_col].values[0]
    
    if p_val < 0.001:
        return f"{value:.3f} ***"
    elif p_val < 0.01:
        return f"{value:.3f} **"
    elif p_val < 0.05:
        return f"{value:.3f} *"
    else:
        return f"{value:.3f}"


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate effect of confidence levels on quadrotor performance')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per confidence level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='confidence_results', help='Base name for output files')
    
    args = parser.parse_args()
    
    # Initialize evaluator with specified seed
    evaluator = ConfidenceLevelEvaluator(seed=args.seed)
    
    # Confidence levels to test (92%, 94%, 96%, 98%)
    confidence_levels = [0.92, 0.94, 0.96, 0.98]
    
    print(f"Evaluating confidence levels: {[f'{cl:.0%}' for cl in confidence_levels]}")
    print(f"Running {args.runs} simulations per confidence level...")
    print(f"Using random seed: {args.seed} for reproducibility")
    
    # Run evaluation with identical scenarios for fair comparison
    results_df = evaluator.evaluate_confidence_levels(
        confidence_levels, 
        runs_per_level=args.runs,
        use_identical_scenarios=True
    )
    
    # Save raw results
    results_csv = f"{args.output}_seed{args.seed}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Raw results saved to {results_csv}")
    
    # Create visualizations
    summary_plot = f"{args.output}_summary_seed{args.seed}.png"
    evaluator.visualize_confidence_results(results_df, summary_plot)
    
    trajectory_plot = f"{args.output}_trajectories_seed{args.seed}.png"
    evaluator.plot_trajectory_comparison(results_df, trajectory_plot)
    
    # Print summary statistics with significance tests
    print("\nSummary Statistics:")
    summary = results_df.groupby('confidence_level').agg({
        'success': lambda x: f"{x.mean()*100:.1f}%",
        'steps': 'mean',
        'path_efficiency': 'mean',
        'average_risk': 'mean'
    })
    
    print(summary)
    
    # Print p-values if available
    if hasattr(evaluator, 'p_values_df'):
        print("\nStatistical Significance (compared to 95% confidence):")
        print(evaluator.p_values_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
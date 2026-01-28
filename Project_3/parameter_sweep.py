"""
Parameter sweep module for batch F1 turbulence experiments.
Enables systematic exploration of parameter space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from config import SimulationConfig, create_parameter_sweep
from runner import run_simulation


class ParameterSweep:
    """Manages batch parameter sweep experiments."""
    
    def __init__(self, configs, output_dir="results/sweep"):
        """
        Initialize parameter sweep.
        
        Args:
            configs: List of SimulationConfig objects
            output_dir: Base directory for sweep results
        """
        self.configs = configs
        self.output_dir = Path(output_dir)
        self.results = []
        self.sweep_metadata = {
            'start_time': None,
            'end_time': None,
            'total_runs': len(configs),
            'completed_runs': 0,
            'failed_runs': 0
        }
    
    def run(self, verbose=True, continue_on_error=True):
        """
        Execute all simulations in the sweep.
        
        Args:
            verbose: Whether to print progress
            continue_on_error: Continue sweep if a simulation fails
            
        Returns:
            List of result paths
        """
        self.sweep_metadata['start_time'] = datetime.now().isoformat()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PARAMETER SWEEP: {self.output_dir.name}")
            print(f"Total Configurations: {len(self.configs)}")
            print(f"{'='*70}\n")
        
        result_paths = []
        
        # Use tqdm for progress bar
        iterator = tqdm(self.configs, desc="Running sweep") if verbose else self.configs
        
        for i, config in enumerate(iterator):
            try:
                if verbose and not isinstance(iterator, tqdm):
                    print(f"\n[{i+1}/{len(self.configs)}] Running: {config.get_run_name()}")
                
                # Run simulation
                results = run_simulation(config, verbose=False)
                
                # Save results
                run_output_dir = self.output_dir / config.get_run_name()
                output_path = results.save(run_output_dir)
                result_paths.append(output_path)
                
                # Store results for analysis
                self.results.append({
                    'config': config,
                    'results': results,
                    'path': output_path,
                    'success': True
                })
                
                self.sweep_metadata['completed_runs'] += 1
                
            except Exception as e:
                print(f"\n✗ Error in {config.get_run_name()}: {str(e)}")
                self.sweep_metadata['failed_runs'] += 1
                
                self.results.append({
                    'config': config,
                    'results': None,
                    'path': None,
                    'success': False,
                    'error': str(e)
                })
                
                if not continue_on_error:
                    raise
        
        self.sweep_metadata['end_time'] = datetime.now().isoformat()
        
        # Save sweep summary
        self._save_sweep_summary()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SWEEP COMPLETE")
            print(f"Successful: {self.sweep_metadata['completed_runs']}/{self.sweep_metadata['total_runs']}")
            if self.sweep_metadata['failed_runs'] > 0:
                print(f"Failed: {self.sweep_metadata['failed_runs']}")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'='*70}\n")
        
        return result_paths
    
    def _save_sweep_summary(self):
        """Save summary of entire sweep."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = self.output_dir / "sweep_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.sweep_metadata, f, indent=2)
        
        # Compile results into DataFrame
        df = self._compile_results_dataframe()
        if df is not None and len(df) > 0:
            csv_file = self.output_dir / "sweep_summary.csv"
            df.to_csv(csv_file, index=False)
            
            # Generate comparison plots
            self._generate_comparison_plots(df)
    
    def _compile_results_dataframe(self):
        """Compile all results into a pandas DataFrame."""
        data = []
        
        for result_dict in self.results:
            if not result_dict['success']:
                continue
            
            config = result_dict['config']
            results = result_dict['results']
            stats = results.get_summary_stats()
            
            row = {
                'run_name': config.get_run_name(),
                'reynolds': config.reynolds,
                'ride_height': config.ride_height,
                'wing_type': config.wing_type,
                'mean_drag': stats['mean_drag'],
                'std_drag': stats['std_drag'],
                'mean_lift': stats['mean_lift'],
                'std_lift': stats['std_lift'],
                'downforce_ratio': stats['downforce_ratio'],
                'final_drag': stats['final_drag'],
                'final_lift': stats['final_lift'],
                'runtime_seconds': results.metadata['runtime_seconds']
            }
            data.append(row)
        
        if len(data) == 0:
            return None
        
        return pd.DataFrame(data)
    
    def _generate_comparison_plots(self, df):
        """Generate comparison plots across parameter sweep."""
        
        # Check if we have variation in both parameters
        has_re_variation = df['reynolds'].nunique() > 1
        has_height_variation = df['ride_height'].nunique() > 1
        
        if has_re_variation and has_height_variation:
            self._plot_2d_heatmap(df)
        elif has_re_variation:
            self._plot_reynolds_comparison(df)
        elif has_height_variation:
            self._plot_height_comparison(df)
        
        # Always create force comparison
        self._plot_force_comparison(df)
    
    def _plot_2d_heatmap(self, df):
        """Create 2D heatmap for Re vs Ride Height."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # Prepare pivot tables
        drag_pivot = df.pivot_table(
            values='mean_drag', 
            index='ride_height', 
            columns='reynolds'
        )
        lift_pivot = df.pivot_table(
            values='mean_lift', 
            index='ride_height', 
            columns='reynolds'
        )
        
        # Drag heatmap
        im1 = axes[0].imshow(drag_pivot.values, aspect='auto', cmap='Reds')
        axes[0].set_xticks(range(len(drag_pivot.columns)))
        axes[0].set_xticklabels([f"{int(x)}" for x in drag_pivot.columns])
        axes[0].set_yticks(range(len(drag_pivot.index)))
        axes[0].set_yticklabels([f"{int(x)}" for x in drag_pivot.index])
        axes[0].set_xlabel("Reynolds Number")
        axes[0].set_ylabel("Ride Height")
        axes[0].set_title("Mean Drag Force")
        plt.colorbar(im1, ax=axes[0])
        
        # Lift heatmap
        im2 = axes[1].imshow(lift_pivot.values, aspect='auto', cmap='RdBu_r')
        axes[1].set_xticks(range(len(lift_pivot.columns)))
        axes[1].set_xticklabels([f"{int(x)}" for x in lift_pivot.columns])
        axes[1].set_yticks(range(len(lift_pivot.index)))
        axes[1].set_yticklabels([f"{int(x)}" for x in lift_pivot.index])
        axes[1].set_xlabel("Reynolds Number")
        axes[1].set_ylabel("Ride Height")
        axes[1].set_title("Mean Lift Force (Negative = Downforce)")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_heatmap.png", dpi=150)
        plt.close()
    
    def _plot_reynolds_comparison(self, df):
        """Create comparison plot for Reynolds number sweep."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        
        reynolds_sorted = df.sort_values('reynolds')
        
        # Drag vs Reynolds
        axes[0].errorbar(
            reynolds_sorted['reynolds'], 
            reynolds_sorted['mean_drag'],
            yerr=reynolds_sorted['std_drag'],
            marker='o', linestyle='-', capsize=5
        )
        axes[0].set_xlabel("Reynolds Number")
        axes[0].set_ylabel("Mean Drag Force")
        axes[0].set_title("Drag vs Reynolds Number")
        axes[0].grid(True, alpha=0.3)
        
        # Lift vs Reynolds
        axes[1].errorbar(
            reynolds_sorted['reynolds'], 
            reynolds_sorted['mean_lift'],
            yerr=reynolds_sorted['std_lift'],
            marker='o', linestyle='-', capsize=5, color='blue'
        )
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Reynolds Number")
        axes[1].set_ylabel("Mean Lift Force")
        axes[1].set_title("Lift vs Reynolds Number\n(Negative = Downforce)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "reynolds_comparison.png", dpi=150)
        plt.close()
    
    def _plot_height_comparison(self, df):
        """Create comparison plot for ride height sweep."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        
        height_sorted = df.sort_values('ride_height')
        
        # Drag vs Height
        axes[0].errorbar(
            height_sorted['ride_height'], 
            height_sorted['mean_drag'],
            yerr=height_sorted['std_drag'],
            marker='s', linestyle='-', capsize=5, color='red'
        )
        axes[0].set_xlabel("Ride Height")
        axes[0].set_ylabel("Mean Drag Force")
        axes[0].set_title("Drag vs Ride Height")
        axes[0].grid(True, alpha=0.3)
        
        # Lift vs Height
        axes[1].errorbar(
            height_sorted['ride_height'], 
            height_sorted['mean_lift'],
            yerr=height_sorted['std_lift'],
            marker='s', linestyle='-', capsize=5, color='blue'
        )
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Ride Height")
        axes[1].set_ylabel("Mean Lift Force")
        axes[1].set_title("Lift vs Ride Height\n(Negative = Downforce)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "height_comparison.png", dpi=150)
        plt.close()
    
    def _plot_force_comparison(self, df):
        """Create scatter plot of drag vs lift."""
        plt.figure(figsize=(8, 6), dpi=100)
        
        scatter = plt.scatter(
            df['mean_drag'], 
            df['mean_lift'],
            c=df['reynolds'],
            s=100,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black'
        )
        
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.xlabel("Mean Drag Force")
        plt.ylabel("Mean Lift Force")
        plt.title("Drag vs Lift Across Parameter Space")
        cbar = plt.colorbar(scatter, label="Reynolds Number")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "force_comparison.png", dpi=150)
        plt.close()


def run_parameter_sweep(reynolds_values, ride_height_values, 
                       base_config=None, output_dir="results/sweep",
                       verbose=True):
    """
    Convenience function to run a parameter sweep.
    
    Args:
        reynolds_values: List of Reynolds numbers
        ride_height_values: List of ride heights
        base_config: Base configuration (uses default if None)
        output_dir: Output directory
        verbose: Print progress
        
    Returns:
        ParameterSweep object with results
    """
    configs = create_parameter_sweep(
        reynolds_values=reynolds_values,
        ride_height_values=ride_height_values,
        base_config=base_config
    )
    
    sweep = ParameterSweep(configs, output_dir)
    sweep.run(verbose=verbose)
    
    return sweep

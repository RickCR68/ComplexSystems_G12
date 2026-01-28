"""
Simulation runner module for F1 Turbulence experiments.
Handles single experiment execution and result management.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from pathlib import Path

from lbm_core import LBMSolver
from boundaries import TunnelBoundaries
from analysis import plot_complexity_dashboard
from aerodynamics import calculate_lift_drag, check_ground_effect
from config import SimulationConfig


class SimulationResults:
    """Container for simulation results."""
    
    def __init__(self, config):
        self.config = config
        self.drag_history = []
        self.lift_history = []
        self.step_history = []
        self.final_flow_field = None
        self.final_velocity_field = None
        self.final_rho_field = None
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'runtime_seconds': None
        }
    
    def add_force_data(self, step, drag, lift):
        """Add force measurement to history."""
        self.step_history.append(step)
        self.drag_history.append(drag)
        self.lift_history.append(lift)
    
    def finalize(self, solver, runtime):
        """Store final simulation state."""
        self.final_velocity_field = solver.u.copy()
        self.final_rho_field = solver.rho.copy()
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['runtime_seconds'] = runtime
    
    def get_summary_stats(self):
        """Calculate summary statistics."""
        drag_array = np.array(self.drag_history)
        lift_array = np.array(self.lift_history)
        
        # Exclude initial transient (first 20% of simulation)
        steady_start = len(drag_array) // 5
        drag_steady = drag_array[steady_start:]
        lift_steady = lift_array[steady_start:]
        
        return {
            'mean_drag': float(np.mean(drag_steady)),
            'std_drag': float(np.std(drag_steady)),
            'mean_lift': float(np.mean(lift_steady)),
            'std_lift': float(np.std(lift_steady)),
            'min_lift': float(np.min(lift_steady)),
            'max_lift': float(np.max(lift_steady)),
            'downforce_ratio': float(np.sum(lift_steady < 0) / len(lift_steady)),
            'final_drag': float(drag_array[-1]),
            'final_lift': float(lift_array[-1])
        }
    
    def save(self, output_dir):
        """Save all results to disk."""
        # Create output directory
        output_path = Path(output_dir) / self.config.get_run_name()
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to: {output_path}")
        
        # Save configuration
        config_file = output_path / "config.json"
        self.config.save(config_file)
        
        # Save force history as numpy arrays
        force_file = output_path / "force_history.npz"
        np.savez(
            force_file,
            steps=np.array(self.step_history),
            drag=np.array(self.drag_history),
            lift=np.array(self.lift_history)
        )
        
        # Save summary statistics
        stats = self.get_summary_stats()
        stats_file = output_path / "summary_stats.json"
        with open(stats_file, 'w') as f:
            combined = {**stats, **self.metadata}
            json.dump(combined, f, indent=2)
        
        # Save final flow field
        if self.config.save_flow_field and self.final_velocity_field is not None:
            flow_file = output_path / "final_flow_field.npz"
            np.savez(
                flow_file,
                velocity=self.final_velocity_field,
                density=self.final_rho_field
            )
        
        # Generate and save plots
        if self.config.save_force_history:
            self._save_force_plot(output_path)
        
        if self.config.save_flow_field:
            self._save_flow_plot(output_path)
        
        print(f"✓ Results saved successfully")
        return output_path
    
    def _save_force_plot(self, output_path):
        """Generate and save force history plot."""
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(self.step_history, self.drag_history, 'r-', 
                 label='Drag (Fx)', linewidth=1.5)
        plt.plot(self.step_history, self.lift_history, 'b-', 
                 label='Lift (Fy)', linewidth=1.5)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Simulation Step")
        plt.ylabel("Force (LBM Units)")
        plt.title(f"Aerodynamic Forces - {self.config.get_run_name()}\n"
                  f"Re={self.config.reynolds}, Ride Height={self.config.ride_height}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "force_history.png", dpi=150)
        plt.close()
    
    def _save_flow_plot(self, output_path):
        """Generate and save final flow field plot."""
        if self.final_velocity_field is None:
            return
        
        velocity_mag = np.sqrt(
            self.final_velocity_field[:,:,0]**2 + 
            self.final_velocity_field[:,:,1]**2
        )
        
        plt.figure(figsize=(12, 5), dpi=150)
        plt.imshow(velocity_mag, origin='lower', cmap='magma')
        plt.colorbar(label="Flow Velocity |u|")
        plt.title(f"Final Flow State - {self.config.get_run_name()}\n"
                  f"Re={self.config.reynolds}, Ride Height={self.config.ride_height}")
        plt.tight_layout()
        plt.savefig(output_path / "final_flow.png", dpi=150)
        plt.close()


def run_simulation(config, verbose=True):
    """
    Run a single simulation with the given configuration.
    
    Args:
        config: SimulationConfig object
        verbose: Whether to print progress updates
        
    Returns:
        SimulationResults object
    """
    import time
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Simulation: {config.get_run_name()}")
        print(f"Reynolds: {config.reynolds}, Ride Height: {config.ride_height}")
        print(f"Domain: {config.nx}x{config.ny}, Steps: {config.total_steps}")
        print(f"{'='*60}\n")
    
    # Initialize solver
    solver = LBMSolver(
        config.nx, 
        config.ny, 
        config.reynolds, 
        config.u_inlet,
        config.cs_smag
    )
    
    # Setup boundaries
    bounds = TunnelBoundaries(config.nx, config.ny)
    bounds.add_ground(type=config.ground_type)
    
    # Add wing geometry
    if config.wing_type == "triangle":
        bounds.add_f1_wing_proxy(
            x_pos=config.wing_x_pos,
            height=config.ride_height,
            length=config.wing_length,
            slope=config.wing_slope
        )
    elif config.wing_type == "reverse_triangle":
        bounds.add_reverse_triangle(
            x_pos=config.wing_x_pos,
            height=config.ride_height,
            length=config.wing_length,
            slope=config.wing_slope
        )
    
    # Initialize results container
    results = SimulationResults(config)
    
    # Main simulation loop
    for step in range(config.total_steps + 1):
        # Physics step
        solver.collide_and_stream(bounds.mask)
        bounds.apply_inlet_outlet(solver)
        
        # Monitor aerodynamic forces
        if step % config.monitor_interval == 0:
            fx, fy = calculate_lift_drag(solver, bounds,
                                         x_start=config.wing_x_pos,
                                         x_end=config.wing_x_pos + config.wing_length,
                                         y_start=config.ride_height,
                                         y_end=config.ride_height + 25)
            results.add_force_data(step, fx, fy)
            
            if verbose:
                check_ground_effect(fx, fy)
                print(f"Step {step}/{config.total_steps}")
        
        # Display complexity dashboard
        if config.save_dashboard and step > 0 and step % config.dashboard_interval == 0:
            if verbose:
                print(f"   --> Generating complexity dashboard...")
            plot_complexity_dashboard(solver, step)
    
    # Finalize results
    runtime = time.time() - start_time
    results.finalize(solver, runtime)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Print summary stats
        stats = results.get_summary_stats()
        print("Summary Statistics (Steady State):")
        print(f"  Mean Drag:  {stats['mean_drag']:.4f} ± {stats['std_drag']:.4f}")
        print(f"  Mean Lift:  {stats['mean_lift']:.4f} ± {stats['std_lift']:.4f}")
        print(f"  Downforce Ratio: {stats['downforce_ratio']*100:.1f}%")
        print()
    
    return results


def run_and_save(config, output_dir=None, verbose=True):
    """
    Run simulation and automatically save results.
    
    Args:
        config: SimulationConfig object
        output_dir: Directory to save results (uses config.output_dir if None)
        verbose: Whether to print progress
        
    Returns:
        Path to saved results
    """
    results = run_simulation(config, verbose=verbose)
    
    if output_dir is None:
        output_dir = config.output_dir
    
    output_path = results.save(output_dir)
    
    return output_path

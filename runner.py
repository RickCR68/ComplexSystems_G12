"""
Simulation Runner Module

High-level interface for running LBM simulations:
- Single runs with monitoring
- Parameter sweeps
- Progress reporting
- Automatic result saving

This module provides the `run_simulation` function used by the notebook
and parameter sweep functionality.
"""

import numpy as np
import time
from typing import Optional, Callable, Tuple
from pathlib import Path

from lbm_core import LBMSolver
from geometry import TunnelBoundaries, GeometryConfig, GeometryType
from aerodynamics import calculate_forces_mem, ForceMonitor, interpret_f1_forces
from analysis import (compute_flow_statistics, compute_energy_spectrum, 
                     ConvergenceMonitor, analyze_spectral_slope)
from visualization import (create_summary_figure, AnimationBuilder,
                          plot_velocity_field, plot_vorticity_field)
from experiment import (SimulationConfig, SimulationResults, 
                       ExperimentManager, ParameterSweep)


def run_simulation(config: SimulationConfig, 
                  exp_dir: Optional[Path] = None,
                  progress_callback: Optional[Callable] = None,
                  verbose: bool = True) -> SimulationResults:
    """
    Run a complete LBM simulation.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    exp_dir : Path, optional
        Directory to save results. If None, results are not saved.
    progress_callback : callable, optional
        Called with (step, total_steps) for progress reporting
    verbose : bool
        Print progress messages
        
    Returns
    -------
    results : SimulationResults
        Complete simulation results
    """
    start_time = time.time()
    
    if verbose:
        print(f"=" * 50)
        print(f"Starting simulation: {config.name}")
        print(f"  Geometry: {config.geometry_type}")
        print(f"  Reynolds: {config.reynolds}")
        print(f"  Ride Height: {config.ride_height}")
        print(f"  Steps: {config.total_steps}")
        print(f"=" * 50)
    
    # === SETUP ===
    
    # Create solver
    solver = LBMSolver(
        nx=config.nx,
        ny=config.ny,
        reynolds=config.reynolds,
        u_inlet=config.u_inlet,
        turbulence_model=config.turbulence_model,
        cs_smag=config.cs_smag,
        effective_re_target=config.effective_re_target
    )
    
    # Create geometry
    geom_config = GeometryConfig(
        geometry_type=GeometryType(config.geometry_type),
        ride_height=config.ride_height,
        scale=config.geometry_scale,
        x_position=config.geometry_x_position,
        angle_of_attack=config.angle_of_attack,
        ground_type=config.ground_type
    )
    bounds = TunnelBoundaries(config.nx, config.ny, geom_config)
    bounds.build()
    
    # Monitors
    force_monitor = ForceMonitor(window_size=config.convergence_window)
    convergence_monitor = ConvergenceMonitor(
        window_size=config.convergence_window,
        check_interval=config.output_interval
    )
    
    # Animation builder (if enabled)
    anim_builder = None
    if config.save_animation:
        anim_builder = AnimationBuilder(solver, bounds)
    
    # === MAIN LOOP ===
    
    converged = False
    convergence_step = 0
    
    for step in range(config.total_steps):
        # Simulation step
        solver.collide_and_stream(bounds.mask)
        bounds.apply_inlet_outlet(solver)
        
        # Moving ground (if configured)
        if config.ground_type == "moving":
            bounds.apply_moving_ground(solver, config.u_inlet)
        
        # Record at intervals
        if step % config.output_interval == 0:
            # Forces
            forces = calculate_forces_mem(solver, bounds)
            force_monitor.record(step, forces)
            
            # Flow statistics
            stats = compute_flow_statistics(solver, bounds.mask)
            convergence_monitor.record(step, stats)
            
            # Animation frame
            if anim_builder is not None and step % config.animation_interval == 0:
                anim_builder.capture_frame((forces.drag, forces.lift))
            
            # Progress
            if progress_callback:
                progress_callback(step, config.total_steps)
            elif verbose and step % (config.output_interval * 10) == 0:
                print(f"  Step {step:>6}/{config.total_steps} | "
                      f"Drag: {forces.drag:>8.4f} | Lift: {forces.lift:>8.4f}")
            
            # Check convergence
            if not converged:
                is_conv, metrics = convergence_monitor.is_converged(
                    tolerance=config.convergence_tolerance
                )
                if is_conv:
                    converged = True
                    convergence_step = step
                    if verbose:
                        print(f"  >>> Converged at step {step}")
    
    # === POST-PROCESSING ===
    
    elapsed = time.time() - start_time
    
    # Final statistics
    mean_forces = force_monitor.get_mean_forces()
    std_drag, std_lift = force_monitor.get_std_forces()
    final_stats = compute_flow_statistics(solver, bounds.mask)
    
    # Create results object
    results = SimulationResults(
        config=config,
        mean_drag=mean_forces.drag,
        mean_lift=mean_forces.lift,
        mean_cd=mean_forces.cd,
        mean_cl=mean_forces.cl,
        std_drag=std_drag,
        std_lift=std_lift,
        converged=converged,
        convergence_step=convergence_step,
        final_step=config.total_steps,
        max_velocity=final_stats.max_velocity,
        kinetic_energy=final_stats.kinetic_energy,
        enstrophy=final_stats.enstrophy,
        wall_time_seconds=elapsed,
        steps_per_second=config.total_steps / elapsed
    )
    
    if verbose:
        print("\n" + results.summary_string())
        interpret_f1_forces(mean_forces, verbose=True)
    
    # === SAVE RESULTS ===
    
    if exp_dir is not None:
        exp_dir = Path(exp_dir)
        manager = ExperimentManager(exp_dir.parent)
        
        # Save data
        manager.save_results(exp_dir, results, solver, bounds, force_monitor)
        
        # Generate and save figures
        import matplotlib.pyplot as plt
        
        # Summary figure
        k, E_k, fft_2d = compute_energy_spectrum(
            solver.u[:, :, 0], solver.u[:, :, 1]
        )
        fig = create_summary_figure(solver, bounds, force_monitor, 
                                   energy_data=(k, E_k, fft_2d))
        manager.save_figure(exp_dir, fig, "summary")
        plt.close(fig)
        
        # Individual field plots
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_velocity_field(solver, bounds, ax=ax)
        manager.save_figure(exp_dir, fig, "velocity")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_vorticity_field(solver, bounds, ax=ax)
        manager.save_figure(exp_dir, fig, "vorticity")
        plt.close(fig)
        
        # Animation
        if anim_builder is not None and len(anim_builder.frames_data) > 0:
            anim_builder.save_animation(
                str(exp_dir / "figures" / "animation.gif"),
                field='vorticity'
            )
    
    return results


def run_parameter_sweep(base_config: SimulationConfig,
                       sweep_params: dict,
                       results_dir: str = "results",
                       verbose: bool = True) -> Tuple[ParameterSweep, list]:
    """
    Run parameter sweep experiment.
    
    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration
    sweep_params : dict
        Parameters to sweep: {param_name: [values]}
    results_dir : str
        Base directory for results
    verbose : bool
        Print progress
        
    Returns
    -------
    sweep : ParameterSweep
        The sweep object (for analysis)
    results : list of SimulationResults
        All results
        
    Example
    -------
    >>> config = SimulationConfig(reynolds=1000, total_steps=5000)
    >>> sweep_params = {
    ...     'ride_height': [3, 5, 7, 10, 15],
    ...     'reynolds': [500, 1000, 2000]
    ... }
    >>> sweep, results = run_parameter_sweep(config, sweep_params)
    """
    manager = ExperimentManager(results_dir)
    sweep = ParameterSweep(base_config, manager)
    
    for param_name, values in sweep_params.items():
        sweep.add_parameter(param_name, values)
    
    def progress(idx, total, name):
        if verbose:
            print(f"\n[{idx+1}/{total}] {name}")
    
    def run_func(cfg, exp_dir):
        return run_simulation(cfg, exp_dir, verbose=False)
    
    results = sweep.run(run_func, progress_callback=progress if verbose else None)
    
    return sweep, results


def quick_run(reynolds: float = 1000,
             ride_height: int = 5,
             geometry: str = "f1_wing_simple",
             steps: int = 5000,
             save: bool = False,
             **kwargs):
    """
    Quick simulation run for testing.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number
    ride_height : int
        Ground clearance
    geometry : str
        Geometry type
    steps : int
        Simulation steps
    save : bool
        Save results
    **kwargs
        Additional config parameters
        
    Returns
    -------
    results : SimulationResults
    solver : LBMSolver (if save=False)
    bounds : TunnelBoundaries (if save=False)
    """
    from experiment import quick_config
    
    config = quick_config(
        reynolds=reynolds,
        ride_height=ride_height,
        geometry=geometry,
        steps=steps,
        **kwargs
    )
    
    exp_dir = None
    if save:
        manager = ExperimentManager("results")
        exp_dir = manager.create_experiment_dir(config.name)
    
    results = run_simulation(config, exp_dir)
    
    return results


# Convenience function for notebook use
def interactive_run(config: SimulationConfig, 
                   live_plot: bool = True,
                   plot_interval: int = 500):
    """
    Run simulation with live plotting (for Jupyter notebooks).
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    live_plot : bool
        Show live updates in notebook
    plot_interval : int
        Steps between plot updates
        
    Returns
    -------
    solver : LBMSolver
        Solver with final state
    bounds : TunnelBoundaries
        Geometry
    force_monitor : ForceMonitor
        Force history
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    
    # Setup
    solver = LBMSolver(
        nx=config.nx, ny=config.ny,
        reynolds=config.reynolds, u_inlet=config.u_inlet,
        turbulence_model=config.turbulence_model,
        cs_smag=config.cs_smag,
        effective_re_target=config.effective_re_target
    )
    
    geom_config = GeometryConfig(
        geometry_type=GeometryType(config.geometry_type),
        ride_height=config.ride_height,
        scale=config.geometry_scale,
        ground_type=config.ground_type
    )
    bounds = TunnelBoundaries(config.nx, config.ny, geom_config)
    bounds.build()
    
    force_monitor = ForceMonitor()
    
    # Setup live plot
    if live_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plt.ion()
    
    # Main loop
    for step in range(config.total_steps):
        solver.collide_and_stream(bounds.mask)
        bounds.apply_inlet_outlet(solver)
        
        if step % config.output_interval == 0:
            forces = calculate_forces_mem(solver, bounds)
            force_monitor.record(step, forces)
        
        if live_plot and step % plot_interval == 0 and step > 0:
            clear_output(wait=True)
            
            # Update velocity plot
            axes[0].clear()
            vel = solver.get_velocity_magnitude()
            vel_masked = np.ma.masked_where(bounds.mask, vel)
            axes[0].imshow(vel_masked, origin='lower', cmap='magma')
            axes[0].set_title(f'Velocity | Step {step}')
            
            # Update force plot
            axes[1].clear()
            steps_arr, drag, lift, _, _ = force_monitor.get_arrays()
            axes[1].plot(steps_arr, drag, 'r-', label='Drag')
            axes[1].plot(steps_arr, lift, 'b-', label='Lift')
            axes[1].axhline(0, color='gray', linestyle='--')
            axes[1].legend()
            axes[1].set_title('Forces')
            axes[1].set_xlabel('Step')
            
            display(fig)
    
    if live_plot:
        plt.ioff()
        plt.close(fig)
    
    return solver, bounds, force_monitor

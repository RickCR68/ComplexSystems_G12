"""
F1 Ground Effect LBM Simulation Package

A Lattice Boltzmann Method (LBM) simulation framework for studying
aerodynamic ground effect in Formula 1 cars.

Modules:
    lbm_core - Core D2Q9 LBM solver with Smagorinsky turbulence model
    geometry - Obstacle geometries and boundary conditions
    aerodynamics - Force calculations using Momentum Exchange Method
    analysis - Flow diagnostics and turbulence statistics
    visualization - Plotting and animation tools
    experiment - Configuration, saving, and parameter sweeps
    runner - High-level simulation interface
"""

from .lbm_core import LBMSolver, D2Q9
from .geometry import (TunnelBoundaries, GeometryConfig, GeometryType, 
                      create_geometry)
from .aerodynamics import (calculate_forces_mem, ForceMonitor, 
                          interpret_f1_forces, AeroForces)
from .analysis import (compute_flow_statistics, compute_energy_spectrum,
                      compute_vorticity, compute_q_criterion,
                      ConvergenceMonitor, analyze_spectral_slope)
from .visualization import (plot_velocity_field, plot_vorticity_field,
                           plot_pressure_field, plot_force_history,
                           plot_energy_spectrum, create_summary_figure,
                           AnimationBuilder)
from .experiment import (SimulationConfig, SimulationResults,
                        ExperimentManager, ParameterSweep, quick_config)
from .runner import run_simulation, run_parameter_sweep, quick_run

__version__ = "1.0.0"
__author__ = "F1 Simulation Team"

__all__ = [
    # Core
    'LBMSolver', 'D2Q9',
    # Geometry
    'TunnelBoundaries', 'GeometryConfig', 'GeometryType', 'create_geometry',
    # Aerodynamics
    'calculate_forces_mem', 'ForceMonitor', 'interpret_f1_forces', 'AeroForces',
    # Analysis
    'compute_flow_statistics', 'compute_energy_spectrum', 'compute_vorticity',
    'compute_q_criterion', 'ConvergenceMonitor', 'analyze_spectral_slope',
    # Visualization
    'plot_velocity_field', 'plot_vorticity_field', 'plot_pressure_field',
    'plot_force_history', 'plot_energy_spectrum', 'create_summary_figure',
    'AnimationBuilder',
    # Experiment
    'SimulationConfig', 'SimulationResults', 'ExperimentManager',
    'ParameterSweep', 'quick_config',
    # Runner
    'run_simulation', 'run_parameter_sweep', 'quick_run'
]

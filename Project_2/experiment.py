"""
Experiment Management Module

Handles:
- Parameter configuration and validation
- Result saving and loading
- Parameter sweep orchestration
- Reproducibility through configuration files

Directory structure:
    results/
    ├── experiment_name/
    │   ├── config.json          # Full configuration
    │   ├── summary.json         # Key results
    │   ├── force_history.csv    # Time series data
    │   ├── final_state.npz      # Flow field snapshots
    │   └── figures/
    │       ├── velocity.png
    │       ├── vorticity.png
    │       └── summary.png
"""

import json
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import warnings
import itertools


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.
    
    All parameters needed to reproduce a simulation run.
    """
    # Domain
    nx: int = 400
    ny: int = 100
    
    # Physics
    reynolds: float = 1000.0
    u_inlet: float = 0.1
    
    # Turbulence model: 'none', 'smagorinsky', 'smag_wall', 'effective_re'
    turbulence_model: str = "smagorinsky"
    cs_smag: float = 0.1  # Smagorinsky constant
    effective_re_target: Optional[float] = None  # For 'effective_re' model: target Re (e.g., 10_000_000)
    
    # Geometry
    geometry_type: str = "f1_wing_simple"
    ride_height: int = 5
    geometry_scale: float = 1.0
    geometry_x_position: Optional[int] = None
    angle_of_attack: float = 0.0
    ground_type: str = "no_slip"  # "no_slip" or "moving"
    
    # Simulation
    total_steps: int = 5000
    output_interval: int = 100  # Steps between force recordings
    
    # Convergence
    convergence_window: int = 500
    convergence_tolerance: float = 0.02
    
    # Output
    save_flow_fields: bool = True
    save_animation: bool = False
    animation_interval: int = 50  # Steps between animation frames
    
    # Metadata
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        # Check stability constraints
        nu = (self.u_inlet * (self.ny / 2)) / self.reynolds
        tau = 3.0 * nu + 0.5
        
        if tau < 0.51:
            warnings.warn(
                f"Configuration may be unstable: tau={tau:.4f} < 0.51. "
                f"Consider reducing Reynolds number or increasing resolution."
            )
            
        if self.u_inlet > 0.3:
            warnings.warn(
                f"Inlet velocity {self.u_inlet} may cause compressibility errors. "
                f"Consider u_inlet < 0.1"
            )
            
        # Auto-generate name if not provided
        if not self.name:
            self.name = f"sim_Re{self.reynolds:.0f}_rh{self.ride_height}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary."""
        return cls(**d)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class SimulationResults:
    """
    Container for simulation results.
    """
    # Configuration used
    config: SimulationConfig
    
    # Aerodynamic results (converged averages)
    mean_drag: float = 0.0
    mean_lift: float = 0.0
    mean_cd: float = 0.0
    mean_cl: float = 0.0
    std_drag: float = 0.0
    std_lift: float = 0.0
    
    # Convergence info
    converged: bool = False
    convergence_step: int = 0
    final_step: int = 0
    
    # Flow statistics
    max_velocity: float = 0.0
    kinetic_energy: float = 0.0
    enstrophy: float = 0.0
    
    # Timing
    wall_time_seconds: float = 0.0
    steps_per_second: float = 0.0
    
    # Time series (optional, for detailed analysis)
    force_history: Optional[Dict[str, np.ndarray]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        d = {
            'config': self.config.to_dict(),
            'mean_drag': self.mean_drag,
            'mean_lift': self.mean_lift,
            'mean_cd': self.mean_cd,
            'mean_cl': self.mean_cl,
            'std_drag': self.std_drag,
            'std_lift': self.std_lift,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.final_step,
            'max_velocity': self.max_velocity,
            'kinetic_energy': self.kinetic_energy,
            'enstrophy': self.enstrophy,
            'wall_time_seconds': self.wall_time_seconds,
            'steps_per_second': self.steps_per_second,
            'timestamp': self.timestamp
        }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationResults":
        """Create from dictionary."""
        config = SimulationConfig.from_dict(d.pop('config'))
        return cls(config=config, **d)
    
    def summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            f"Simulation Results: {self.config.name}",
            "=" * 50,
            f"Geometry: {self.config.geometry_type}",
            f"Reynolds: {self.config.reynolds:.0f}",
            f"Ride Height: {self.config.ride_height}",
            "",
            "--- Aerodynamic Forces ---",
            f"Mean Drag:     {self.mean_drag:>10.6f} (Cd = {self.mean_cd:.4f})",
            f"Mean Lift:     {self.mean_lift:>10.6f} (Cl = {self.mean_cl:.4f})",
            f"Downforce:     {-self.mean_lift:>10.6f}" if self.mean_lift < 0 else f"LIFT:          {self.mean_lift:>10.6f}",
            f"Std Drag:      {self.std_drag:>10.6f}",
            f"Std Lift:      {self.std_lift:>10.6f}",
            "",
            "--- Convergence ---",
            f"Converged:     {'Yes' if self.converged else 'No'}",
            f"Final Step:    {self.final_step}",
            "",
            "--- Performance ---",
            f"Wall Time:     {self.wall_time_seconds:.1f} s",
            f"Speed:         {self.steps_per_second:.1f} steps/s",
            "=" * 50
        ]
        return "\n".join(lines)


class ExperimentManager:
    """
    Manage experiment directories and results.
    
    Parameters
    ----------
    base_dir : str
        Base directory for all experiments
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_experiment_dir(self, name: str, 
                             overwrite: bool = False) -> Path:
        """
        Create directory for a new experiment.
        
        Parameters
        ----------
        name : str
            Experiment name
        overwrite : bool
            If True, remove existing directory
            
        Returns
        -------
        exp_dir : Path
            Experiment directory path
        """
        exp_dir = self.base_dir / name
        
        if exp_dir.exists():
            if overwrite:
                import shutil
                shutil.rmtree(exp_dir)
            else:
                # Append timestamp to make unique
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_dir = self.base_dir / f"{name}_{timestamp}"
                
        exp_dir.mkdir(parents=True)
        (exp_dir / "figures").mkdir()
        
        return exp_dir
    
    def save_results(self, exp_dir: Path, results: SimulationResults,
                    solver=None, bounds=None, force_monitor=None):
        """
        Save all results to experiment directory.
        
        Parameters
        ----------
        exp_dir : Path
            Experiment directory
        results : SimulationResults
            Results to save
        solver : LBMSolver, optional
            Solver with final state (for flow field snapshots)
        bounds : TunnelBoundaries, optional
            Geometry
        force_monitor : ForceMonitor, optional
            Force history
        """
        exp_dir = Path(exp_dir)
        
        # Save configuration
        results.config.save(exp_dir / "config.json")
        
        # Save summary results
        with open(exp_dir / "summary.json", 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
            
        # Save human-readable summary
        with open(exp_dir / "summary.txt", 'w') as f:
            f.write(results.summary_string())
            
        # Save force history as CSV
        if force_monitor is not None:
            steps, drag, lift, cd, cl = force_monitor.get_arrays()
            if len(steps) > 0:
                history = np.column_stack([steps, drag, lift, cd, cl])
                header = "step,drag,lift,cd,cl"
                np.savetxt(exp_dir / "force_history.csv", history,
                          delimiter=',', header=header, comments='')
                
        # Save flow field snapshots
        if solver is not None and results.config.save_flow_fields:
            np.savez_compressed(
                exp_dir / "final_state.npz",
                u=solver.u,
                rho=solver.rho,
                f=solver.f,
                mask=bounds.mask if bounds else None
            )
            
        print(f"Results saved to {exp_dir}")
        
    def save_figure(self, exp_dir: Path, fig, name: str, 
                   formats: List[str] = ['png', 'pdf']):
        """
        Save figure in multiple formats.
        
        Parameters
        ----------
        exp_dir : Path
            Experiment directory
        fig : matplotlib.figure.Figure
            Figure to save
        name : str
            Base filename (without extension)
        formats : list
            File formats to save
        """
        fig_dir = Path(exp_dir) / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        for fmt in formats:
            filepath = fig_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
    def load_results(self, exp_name: str) -> SimulationResults:
        """
        Load results from experiment directory.
        
        Parameters
        ----------
        exp_name : str
            Experiment name (directory name)
            
        Returns
        -------
        results : SimulationResults
        """
        exp_dir = self.base_dir / exp_name
        
        with open(exp_dir / "summary.json", 'r') as f:
            d = json.load(f)
            
        results = SimulationResults.from_dict(d)
        
        # Load force history if available
        history_file = exp_dir / "force_history.csv"
        if history_file.exists():
            data = np.loadtxt(history_file, delimiter=',', skiprows=1)
            results.force_history = {
                'steps': data[:, 0],
                'drag': data[:, 1],
                'lift': data[:, 2],
                'cd': data[:, 3],
                'cl': data[:, 4]
            }
            
        return results
    
    def list_experiments(self) -> List[str]:
        """List all experiment directories."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def load_all_results(self) -> Dict[str, SimulationResults]:
        """Load results from all experiments."""
        results = {}
        for name in self.list_experiments():
            try:
                results[name] = self.load_results(name)
            except Exception as e:
                warnings.warn(f"Failed to load {name}: {e}")
        return results


class ParameterSweep:
    """
    Orchestrate parameter sweep experiments.
    
    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration (parameters not being swept)
    experiment_manager : ExperimentManager
        Manager for saving results
    """
    
    def __init__(self, base_config: SimulationConfig, 
                 experiment_manager: ExperimentManager):
        self.base_config = base_config
        self.manager = experiment_manager
        self.sweep_params = {}
        self.results = []
        
    def add_parameter(self, name: str, values: List[Any]):
        """
        Add parameter to sweep.
        
        Parameters
        ----------
        name : str
            Parameter name (must match SimulationConfig attribute)
        values : list
            Values to sweep over
        """
        if not hasattr(self.base_config, name):
            raise ValueError(f"Unknown parameter: {name}")
        self.sweep_params[name] = values
        
    def generate_configs(self) -> List[Tuple[str, SimulationConfig]]:
        """
        Generate all configuration combinations.
        
        Returns
        -------
        configs : list of (name, config) tuples
        """
        if not self.sweep_params:
            return [("baseline", self.base_config)]
            
        # Generate all combinations
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        configs = []
        for combo in itertools.product(*param_values):
            # Create config copy
            config_dict = self.base_config.to_dict()
            
            # Apply swept parameters
            name_parts = []
            for pname, pval in zip(param_names, combo):
                config_dict[pname] = pval
                # Create short name for this parameter
                if isinstance(pval, float):
                    name_parts.append(f"{pname[:3]}{pval:.2f}")
                else:
                    name_parts.append(f"{pname[:3]}{pval}")
                    
            config = SimulationConfig.from_dict(config_dict)
            name = "_".join(name_parts)
            config.name = name
            
            configs.append((name, config))
            
        return configs
    
    def run(self, run_simulation_func: Callable, 
           progress_callback: Callable = None) -> List[SimulationResults]:
        """
        Execute parameter sweep.
        
        Parameters
        ----------
        run_simulation_func : callable
            Function that takes (config, exp_dir) and returns SimulationResults
        progress_callback : callable, optional
            Called with (current_idx, total, config_name) for progress reporting
            
        Returns
        -------
        results : list of SimulationResults
        """
        configs = self.generate_configs()
        self.results = []
        
        print(f"Starting parameter sweep: {len(configs)} configurations")
        print(f"Parameters: {list(self.sweep_params.keys())}")
        print("-" * 50)
        
        for idx, (name, config) in enumerate(configs):
            if progress_callback:
                progress_callback(idx, len(configs), name)
            else:
                print(f"\n[{idx+1}/{len(configs)}] Running: {name}")
                
            # Create experiment directory
            exp_dir = self.manager.create_experiment_dir(name)
            
            # Run simulation
            try:
                result = run_simulation_func(config, exp_dir)
                self.results.append(result)
            except Exception as e:
                warnings.warn(f"Simulation {name} failed: {e}")
                continue
                
        print("\n" + "=" * 50)
        print(f"Sweep complete: {len(self.results)}/{len(configs)} successful")
        
        return self.results
    
    def get_results_dataframe(self):
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns
        -------
        df : pandas.DataFrame
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame output")
            
        records = []
        for r in self.results:
            record = {
                'name': r.config.name,
                'reynolds': r.config.reynolds,
                'ride_height': r.config.ride_height,
                'geometry': r.config.geometry_type,
                'mean_drag': r.mean_drag,
                'mean_lift': r.mean_lift,
                'mean_cd': r.mean_cd,
                'mean_cl': r.mean_cl,
                'downforce': -r.mean_lift,
                'converged': r.converged,
                'efficiency': abs(r.mean_lift) / r.mean_drag if r.mean_drag > 0 else np.nan
            }
            records.append(record)
            
        return pd.DataFrame(records)
    
    def plot_sweep_results(self, x_param: str, y_param: str = 'mean_lift',
                          hue_param: str = None, ax=None):
        """
        Plot parameter sweep results.
        
        Parameters
        ----------
        x_param : str
            Parameter for x-axis
        y_param : str
            Result metric for y-axis
        hue_param : str, optional
            Parameter for color grouping
        ax : matplotlib.axes.Axes, optional
            Axis to plot on
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        df = self.get_results_dataframe()
        
        if hue_param and hue_param in df.columns:
            for hue_val, group in df.groupby(hue_param):
                ax.plot(group[x_param], group[y_param], 'o-',
                       label=f'{hue_param}={hue_val}', markersize=8)
            ax.legend()
        else:
            ax.plot(df[x_param], df[y_param], 'o-', markersize=8)
            
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f'{y_param} vs {x_param}')
        ax.grid(True, alpha=0.3)
        
        return ax


def quick_config(reynolds: float = 1000, ride_height: int = 5,
                geometry: str = "f1_wing_simple",
                steps: int = 5000, **kwargs) -> SimulationConfig:
    """
    Convenience function for creating configurations.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number
    ride_height : int
        Ground clearance
    geometry : str
        Geometry type
    steps : int
        Total simulation steps
    **kwargs
        Additional SimulationConfig parameters
        
    Returns
    -------
    config : SimulationConfig
    """
    return SimulationConfig(
        reynolds=reynolds,
        ride_height=ride_height,
        geometry_type=geometry,
        total_steps=steps,
        **kwargs
    )

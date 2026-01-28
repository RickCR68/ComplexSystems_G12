"""
Configuration module for F1 Turbulence Simulation.
Centralizes all simulation parameters for easy experiment management.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class SimulationConfig:
    """Container for all simulation parameters."""
    
    # Domain Configuration
    nx: int = 400
    ny: int = 100
    
    # Physics Parameters
    reynolds: float = 10000
    u_inlet: float = 0.1
    
    # Geometry Parameters
    ride_height: int = 5  # Distance from ground to wing base
    wing_x_pos: int = 50
    wing_length: int = 30
    wing_slope: float = 0.5
    wing_type: str = "triangle"  # "triangle" or "reverse_triangle"
    
    # Ground Configuration
    ground_type: str = "no_slip"  # "no_slip" or "slip"
    
    # Simulation Control
    total_steps: int = 10000
    monitor_interval: int = 100  # Steps between force calculations
    dashboard_interval: int = 500  # Steps between complexity plots
    
    # Turbulence Model
    cs_smag: float = 0.15  # Smagorinsky constant
    
    # Output Control
    save_flow_field: bool = True
    save_force_history: bool = True
    save_dashboard: bool = False
    output_dir: str = "results"
    
    # Experiment Metadata
    experiment_name: str = "f1_turbulence"
    run_id: Optional[str] = None
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_run_name(self):
        """Generate a descriptive name for this run."""
        if self.run_id:
            return self.run_id
        return f"Re{int(self.reynolds)}_h{self.ride_height}_{self.wing_type}"


def create_parameter_sweep(
    reynolds_values=None,
    ride_height_values=None,
    base_config=None
):
    """
    Generate a list of configurations for parameter sweep.
    
    Args:
        reynolds_values: List of Reynolds numbers to test
        ride_height_values: List of ride heights to test
        base_config: Base configuration to modify
        
    Returns:
        List of SimulationConfig objects
    """
    if base_config is None:
        base_config = SimulationConfig()
    
    if reynolds_values is None:
        reynolds_values = [5000, 10000, 15000]
    
    if ride_height_values is None:
        ride_height_values = [3, 5, 7, 10]
    
    configs = []
    
    for re in reynolds_values:
        for height in ride_height_values:
            # Create a new config based on base
            config_dict = base_config.to_dict()
            config_dict['reynolds'] = re
            config_dict['ride_height'] = height
            config_dict['run_id'] = f"Re{int(re)}_h{height}"
            
            configs.append(SimulationConfig.from_dict(config_dict))
    
    return configs


# Predefined experiment configurations
QUICK_TEST = SimulationConfig(
    nx=200,
    ny=50,
    reynolds=5000,
    total_steps=1000,
    monitor_interval=100,
    dashboard_interval=500,
    experiment_name="quick_test"
)

STANDARD_RUN = SimulationConfig(
    nx=400,
    ny=100,
    reynolds=10000,
    total_steps=10000,
    monitor_interval=100,
    dashboard_interval=500,
    experiment_name="standard_run"
)

HIGH_RESOLUTION = SimulationConfig(
    nx=800,
    ny=200,
    reynolds=20000,
    total_steps=15000,
    monitor_interval=200,
    dashboard_interval=1000,
    experiment_name="high_res"
)

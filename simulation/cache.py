"""
Results caching and persistence for LBM simulations.

Saves and loads simulation results using JSON-backed dataclasses to avoid
recomputing expensive simulations.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Tuple


@dataclass
class SimulationResult:
    """
    Stores simulation output and metadata.
    
    Designed to be serializable to/from JSON.
    """
    scenario_name: str
    grid_size_x: int
    grid_size_y: int
    viscosity: float
    inlet_velocity: float
    num_iterations: int
    iteration_count: int  # Actual iterations completed
    
    # Field data (stored as lists for JSON compatibility)
    velocity_x: list  # Shape (ny, nx)
    velocity_y: list  # Shape (ny, nx)
    vorticity: list   # Shape (ny, nx)
    density: list     # Shape (ny, nx)
    
    # Metadata
    reynolds_number: float
    drag_coefficient: Optional[float] = None
    lift_coefficient: Optional[float] = None
    max_velocity: Optional[float] = None
    convergence_detected: bool = False
    timestamp: str = ""
    description: str = ""
    
    def to_json(self, path: Path) -> None:
        """
        Save result to JSON file.
        
        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "SimulationResult":
        """
        Load result from JSON file.
        
        Args:
            path: Input file path
        
        Returns:
            SimulationResult instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def get_velocity_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert stored velocity lists to NumPy arrays.
        
        Returns:
            (velocity_x, velocity_y) as numpy arrays
        """
        return np.array(self.velocity_x), np.array(self.velocity_y)
    
    def get_vorticity_array(self) -> np.ndarray:
        """
        Convert stored vorticity list to NumPy array.
        
        Returns:
            Vorticity field as numpy array
        """
        return np.array(self.vorticity)
    
    def get_density_array(self) -> np.ndarray:
        """
        Convert stored density list to NumPy array.
        
        Returns:
            Density field as numpy array
        """
        return np.array(self.density)


class SimulationCache:
    """
    Manages caching of simulation results.
    
    Provides simple interface to save/load simulations without recomputing.
    """
    
    def __init__(self, cache_dir: Path = Path("results")):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_filename(self, scenario_name: str, suffix: str = "") -> Path:
        """
        Generate cache filename from scenario name and timestamp.
        
        Args:
            scenario_name: Name of scenario (e.g., "poiseuille")
            suffix: Additional suffix for cache file
        
        Returns:
            Path object for cache file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name}_{timestamp}{suffix}.json"
        return self.cache_dir / filename
    
    def save_result(self, result: SimulationResult, custom_name: Optional[str] = None) -> Path:
        """
        Save simulation result to cache.
        
        Args:
            result: SimulationResult object
            custom_name: Custom filename (without extension); overrides default naming
        
        Returns:
            Path to saved file
        """
        if custom_name:
            path = self.cache_dir / f"{custom_name}.json"
        else:
            path = self.get_cache_filename(result.scenario_name)
        
        result.timestamp = datetime.now().isoformat()
        result.to_json(path)
        
        return path
    
    def load_latest(self, scenario_name: str) -> Optional[SimulationResult]:
        """
        Load most recent cached result for a scenario.
        
        Args:
            scenario_name: Name of scenario to load
        
        Returns:
            SimulationResult or None if no cache found
        """
        # Find all cache files for this scenario
        pattern = f"{scenario_name}_*.json"
        cache_files = sorted(self.cache_dir.glob(pattern))
        
        if not cache_files:
            return None
        
        # Return the most recent
        return SimulationResult.from_json(cache_files[-1])
    
    def load_by_path(self, path: Path) -> SimulationResult:
        """
        Load result from specific path.
        
        Args:
            path: Path to cache file
        
        Returns:
            SimulationResult
        """
        return SimulationResult.from_json(path)
    
    def list_cached_scenarios(self) -> Dict[str, list]:
        """
        List all cached scenarios and their files.
        
        Returns:
            Dict mapping scenario names to list of file paths
        """
        scenarios = {}
        for json_file in self.cache_dir.glob("*.json"):
            scenario_name = json_file.stem.rsplit('_', 2)[0]  # Extract before timestamp
            if scenario_name not in scenarios:
                scenarios[scenario_name] = []
            scenarios[scenario_name].append(json_file)
        
        return scenarios
    
    def clear_cache(self, scenario_name: Optional[str] = None) -> int:
        """
        Clear cached results.
        
        Args:
            scenario_name: Scenario to clear; if None, clears all
        
        Returns:
            Number of files deleted
        """
        count = 0
        if scenario_name:
            pattern = f"{scenario_name}_*.json"
            for f in self.cache_dir.glob(pattern):
                f.unlink()
                count += 1
        else:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                count += 1
        
        return count


__all__ = ["SimulationResult", "SimulationCache"]

"""Lean configuration dataclasses for the LBM simulation."""

from dataclasses import dataclass, asdict, field
from typing import List
import json
from pathlib import Path


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = None  # For circles
    type: str = "circle"  # "circle" or "triangle"
    width: float = None  # For triangles (base width)
    height: float = None  # For triangles (height)
    angle: float = 0  # Rotation angle in degrees (0 = point right)


@dataclass
class SimulationConfig:
    scenario_name: str = "default"
    description: str = ""
    grid_size_x: int = 256
    grid_size_y: int = 128
    viscosity: float = 0.01
    inlet_velocity: float = 0.1
    density_ref: float = 1.0
    num_iterations: int = 5000
    save_interval: int = 500
    obstacles: List[Obstacle] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> "SimulationConfig":
        with open(path) as f:
            data = json.load(f)
        data["obstacles"] = [Obstacle(**obs) for obs in data.get("obstacles", [])]
        return cls(**data)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["obstacles"] = [asdict(obs) for obs in self.obstacles]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["obstacles"] = [asdict(obs) for obs in self.obstacles]
        return data

    @property
    def reynolds_number(self) -> float:
        return self.inlet_velocity * self.grid_size_y / self.viscosity

    @property
    def relaxation_time(self) -> float:
        return 3.0 * self.viscosity + 0.5


__all__ = ["SimulationConfig", "Obstacle"]

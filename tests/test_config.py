"""Tests for the lean SimulationConfig dataclass."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from config.parameters import SimulationConfig, Obstacle


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.grid_size_x == 256
        assert cfg.grid_size_y == 128
        assert cfg.viscosity == 0.01
        assert cfg.inlet_velocity == 0.1

    def test_obstacles(self):
        cfg = SimulationConfig(obstacles=[Obstacle(x=10, y=5, radius=2)])
        assert len(cfg.obstacles) == 1
        assert cfg.obstacles[0].radius == 2

    def test_reynolds_and_tau(self):
        cfg = SimulationConfig(grid_size_y=64, inlet_velocity=0.2, viscosity=0.02)
        assert np.isclose(cfg.reynolds_number, 0.2 * 64 / 0.02)
        assert np.isclose(cfg.relaxation_time, 3 * 0.02 + 0.5)

    def test_json_roundtrip(self):
        cfg = SimulationConfig(
            scenario_name="demo",
            grid_size_x=100,
            grid_size_y=50,
            viscosity=0.02,
            inlet_velocity=0.15,
            obstacles=[Obstacle(x=20, y=25, radius=5)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            cfg.to_json(path)
            loaded = SimulationConfig.from_json(path)
            assert loaded.scenario_name == "demo"
            assert loaded.grid_size_x == 100
            assert len(loaded.obstacles) == 1
            assert loaded.obstacles[0].x == 20


class TestObstacle:
    def test_creation(self):
        obs = Obstacle(x=1.0, y=2.0, radius=3.0)
        assert obs.x == 1.0
        assert obs.radius == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

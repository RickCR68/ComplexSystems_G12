"""Lean tests for the combined LBM core."""

import numpy as np
import pytest

from config.parameters import SimulationConfig, Obstacle
from simulation.core import LBMSimulation


class TestLBMSimulation:
    @pytest.fixture
    def sim(self):
        cfg = SimulationConfig(grid_size_x=64, grid_size_y=32, viscosity=0.01)
        return LBMSimulation(cfg)

    def test_initialization(self, sim):
        assert sim.nx == 64
        assert sim.ny == 32
        assert sim.state.f.shape == (9, 32, 64)
        assert np.isclose(sim.tau, 3 * 0.01 + 0.5)

    def test_step_runs(self, sim):
        sim.step(inlet_velocity=0.05)
        rho, ux, uy = sim.state.rho, sim.state.ux, sim.state.uy
        assert rho.shape == (32, 64)
        assert ux.shape == (32, 64)
        assert uy.shape == (32, 64)

    def test_obstacle_mask(self):
        cfg = SimulationConfig(
            grid_size_x=64,
            grid_size_y=32,
            obstacles=[Obstacle(x=20, y=16, radius=4)],
        )
        sim = LBMSimulation(cfg)
        assert sim.mask[16, 20] == 1
        assert np.sum(sim.mask) > 0

    def test_vorticity(self, sim):
        # Run a few steps to generate velocity gradients
        sim.run(inlet_velocity=0.05, num_steps=5)
        vort = sim.compute_vorticity()
        assert vort.shape == (sim.ny, sim.nx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

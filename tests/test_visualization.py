"""
Unit tests for visualization utilities.

Tests plotting and field computation functions.
"""

import pytest
import numpy as np
from simulation.visualization import Visualizer


class TestVisualizer:
    """Tests for visualization utilities."""
    
    @pytest.fixture
    def velocity_fields(self):
        """Create test velocity fields."""
        ny, nx = 64, 128
        ux = 0.1 * np.ones((ny, nx), dtype=np.float32)
        uy = 0.01 * np.sin(2 * np.pi * np.arange(nx) / nx).reshape(1, -1) * np.ones((ny, 1))
        
        return ux, uy
    
    @pytest.fixture
    def mask(self):
        """Create test obstacle mask."""
        mask = np.zeros((64, 128), dtype=np.int32)
        yy, xx = np.ogrid[:64, :128]
        circle = (xx - 64)**2 + (yy - 32)**2 <= 8**2
        mask[circle] = 1
        return mask
    
    def test_vorticity_computation(self, velocity_fields):
        """Test vorticity computation from velocity field."""
        ux, uy = velocity_fields
        
        vort = Visualizer.compute_vorticity(ux, uy)
        
        # Check shape
        assert vort.shape == ux.shape
        
        # Vorticity should be non-zero due to velocity gradients
        assert not np.allclose(vort, 0)
    
    def test_vorticity_zero_uniform_flow(self):
        """Test that uniform flow has zero vorticity."""
        ny, nx = 64, 128
        ux = 0.1 * np.ones((ny, nx))
        uy = np.zeros((ny, nx))
        
        vort = Visualizer.compute_vorticity(ux, uy)
        
        # Vorticity should be zero everywhere for uniform flow
        assert np.allclose(vort, 0, atol=1e-10)
    
    def test_plot_velocity_magnitude(self, velocity_fields):
        """Test velocity magnitude plotting."""
        ux, uy = velocity_fields
        
        fig, ax = Visualizer.plot_velocity_magnitude(ux, uy)
        
        # Check that figure was created
        assert fig is not None
        assert ax is not None
    
    def test_plot_vorticity(self, velocity_fields):
        """Test vorticity plotting."""
        ux, uy = velocity_fields
        vort = Visualizer.compute_vorticity(ux, uy)
        
        fig, ax = Visualizer.plot_vorticity(vort)
        
        assert fig is not None
        assert ax is not None
    
    def test_plot_velocity_quiver(self, velocity_fields):
        """Test quiver plot."""
        ux, uy = velocity_fields
        
        fig, ax = Visualizer.plot_velocity_quiver(ux, uy, stride=4)
        
        assert fig is not None
        assert ax is not None
    
    def test_plot_streamlines(self, velocity_fields):
        """Test streamline plot."""
        ux, uy = velocity_fields
        
        fig, ax = Visualizer.plot_streamlines(ux, uy)
        
        assert fig is not None
        assert ax is not None
    
    def test_plot_with_mask(self, velocity_fields, mask):
        """Test that plotting functions work with obstacle mask."""
        ux, uy = velocity_fields
        
        # Should not raise error
        fig1, ax1 = Visualizer.plot_velocity_magnitude(ux, uy, mask=mask)
        fig2, ax2 = Visualizer.plot_velocity_quiver(ux, uy, mask=mask)
        
        assert fig1 is not None
        assert fig2 is not None
    
    def test_plot_velocity_profile(self, velocity_fields):
        """Test velocity profile extraction."""
        ux, uy = velocity_fields
        
        fig, ax = Visualizer.plot_velocity_profile(ux, y_slice=32)
        
        assert fig is not None
        assert ax is not None
    
    def test_plot_velocity_profile_with_analytical(self, velocity_fields):
        """Test velocity profile with analytical comparison."""
        ux, uy = velocity_fields
        analytical = 0.11 * np.ones_like(ux[0, :])
        
        fig, ax = Visualizer.plot_velocity_profile(ux, y_slice=32, analytical=analytical)
        
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

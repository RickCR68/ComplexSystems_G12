"""
Basic tests for LBM implementation.
Run with: pytest test_lbm.py
"""

import numpy as np
import sys
sys.path.append('../src')

from lbm_core import LBMSimulation, D2Q9Lattice
from validation import poiseuille_analytical, compute_l2_error
from geometry import create_cylinder


def test_lattice_initialization():
    """Test that D2Q9 lattice is properly initialized."""
    lattice = D2Q9Lattice()
    
    # Check number of velocities
    assert lattice.num_velocities == 9
    
    # Check weights sum to 1
    assert np.isclose(np.sum(lattice.w), 1.0)
    
    # Check opposite directions
    for i in range(9):
        opp = lattice.opposite[i]
        # c[i] + c[opposite[i]] should be close to zero
        assert np.allclose(lattice.c[:, i] + lattice.c[:, opp], 0)


def test_simulation_initialization():
    """Test simulation initialization."""
    sim = LBMSimulation(nx=100, ny=50, reynolds_number=100, u_max=0.1)
    
    assert sim.nx == 100
    assert sim.ny == 50
    assert sim.Re == 100
    assert sim.iteration == 0
    
    # Check distribution functions are initialized
    assert sim.f.shape == (9, 100, 50)
    assert sim.rho.shape == (100, 50)
    assert sim.u.shape == (2, 100, 50)


def test_mass_conservation():
    """Test that mass is conserved during simulation."""
    sim = LBMSimulation(nx=100, ny=50, reynolds_number=100, u_max=0.1)
    sim.initialize_equilibrium(rho_init=1.0)
    
    initial_mass = np.sum(sim.rho)
    
    # Run a few steps
    for _ in range(100):
        sim.step()
    
    final_mass = np.sum(sim.rho)
    
    # Mass should be conserved (within numerical precision)
    assert np.isclose(initial_mass, final_mass, rtol=1e-6)


def test_poiseuille_validation():
    """Test that Poiseuille flow converges to analytical solution."""
    nx, ny = 200, 50
    Re = 100
    u_max = 0.1
    
    sim = LBMSimulation(nx=nx, ny=ny, reynolds_number=Re, u_max=u_max)
    
    # Initialize with parabolic profile
    y = np.arange(ny)
    y_center = (ny - 1) / 2.0
    u_init = u_max * (1 - ((y - y_center) / y_center)**2)
    u_field = np.zeros((2, nx, ny))
    u_field[0, :, :] = u_init[np.newaxis, :]
    sim.initialize_equilibrium(rho_init=1.0, u_init=u_field)
    
    # Run simulation
    for _ in range(2000):
        sim.step()
    
    # Get numerical solution
    u_numerical = sim.u[0, nx//2, :]
    
    # Get analytical solution
    u_analytical = poiseuille_analytical(ny, u_max)
    
    # Compute error
    error = compute_l2_error(u_numerical, u_analytical)
    
    # Error should be small (less than 1% of u_max)
    assert error < 0.01 * u_max


def test_cylinder_geometry():
    """Test cylinder geometry creation."""
    nx, ny = 200, 100
    radius = 10
    center = (50, 50)
    
    cylinder = create_cylinder(nx, ny, center, radius)
    
    # Check that cylinder is boolean
    assert cylinder.dtype == bool
    
    # Check shape
    assert cylinder.shape == (nx, ny)
    
    # Check that center is inside cylinder
    assert cylinder[center[0], center[1]] == True
    
    # Check that a point far away is outside
    assert cylinder[0, 0] == False
    
    # Rough check on area (should be approximately pi*r^2)
    area = np.sum(cylinder)
    expected_area = np.pi * radius**2
    assert np.isclose(area, expected_area, rtol=0.2)  # 20% tolerance


def test_equilibrium_calculation():
    """Test equilibrium distribution calculation."""
    sim = LBMSimulation(nx=100, ny=50, reynolds_number=100, u_max=0.1)
    
    # Set uniform density and velocity
    rho = np.ones((100, 50))
    u = np.zeros((2, 100, 50))
    u[0] = 0.1  # Uniform flow in x-direction
    
    sim._compute_equilibrium(rho, u)
    
    # Check that f_eq sums to rho
    rho_from_feq = np.sum(sim.f_eq, axis=0)
    assert np.allclose(rho_from_feq, rho)
    
    # Check that momentum is correct
    c = sim.lattice.c
    u_from_feq = np.sum(c[0, :, np.newaxis, np.newaxis] * sim.f_eq, axis=0) / rho
    v_from_feq = np.sum(c[1, :, np.newaxis, np.newaxis] * sim.f_eq, axis=0) / rho
    
    assert np.allclose(u_from_feq, u[0])
    assert np.allclose(v_from_feq, u[1])


if __name__ == "__main__":
    # Run tests manually
    print("Running tests...")
    
    test_lattice_initialization()
    print("✓ Lattice initialization")
    
    test_simulation_initialization()
    print("✓ Simulation initialization")
    
    test_mass_conservation()
    print("✓ Mass conservation")
    
    test_cylinder_geometry()
    print("✓ Cylinder geometry")
    
    test_equilibrium_calculation()
    print("✓ Equilibrium calculation")
    
    print("\n⚠ Skipping Poiseuille validation (takes longer)")
    print("  Run with: pytest test_lbm.py::test_poiseuille_validation")
    
    print("\nAll quick tests passed! ✓")

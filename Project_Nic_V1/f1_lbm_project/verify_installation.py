#!/usr/bin/env python3
"""
Quick verification script for F1 LBM project.
Run this to verify installation and basic functionality.
"""

import sys
import numpy as np

print("="*60)
print("F1 LBM Project - Quick Verification")
print("="*60)

# Add src to path
sys.path.append('src')

try:
    print("\n1. Testing imports...")
    from lbm_core import LBMSimulation, D2Q9Lattice
    from boundary_conditions import BoundaryConditions
    from geometry import create_cylinder
    from validation import poiseuille_analytical
    from analysis import ConvergenceMonitor
    from visualization import plot_velocity_field
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

try:
    print("\n2. Testing lattice initialization...")
    lattice = D2Q9Lattice()
    assert lattice.num_velocities == 9
    assert np.isclose(np.sum(lattice.w), 1.0)
    print("   ✓ D2Q9 lattice initialized correctly")
except AssertionError as e:
    print(f"   ✗ Lattice test failed: {e}")
    sys.exit(1)

try:
    print("\n3. Testing simulation creation...")
    sim = LBMSimulation(nx=100, ny=50, reynolds_number=100, u_max=0.1)
    assert sim.nx == 100
    assert sim.ny == 50
    print(f"   ✓ Simulation created: {sim.nx}×{sim.ny}")
    print(f"     Reynolds number: {sim.Re}")
    print(f"     Relaxation time τ: {sim.tau:.4f}")
    print(f"     Kinematic viscosity ν: {sim.nu:.6f}")
except Exception as e:
    print(f"   ✗ Simulation creation failed: {e}")
    sys.exit(1)

try:
    print("\n4. Testing initialization...")
    sim.initialize_equilibrium(rho_init=1.0)
    assert np.allclose(sim.rho, 1.0)
    print("   ✓ Equilibrium initialization successful")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    sys.exit(1)

try:
    print("\n5. Testing LBM iteration...")
    initial_mass = np.sum(sim.rho)
    
    # Run 100 steps
    for i in range(100):
        sim.step()
    
    final_mass = np.sum(sim.rho)
    mass_error = abs(final_mass - initial_mass) / initial_mass
    
    print(f"   ✓ Completed 100 iterations")
    print(f"     Initial mass: {initial_mass:.6f}")
    print(f"     Final mass: {final_mass:.6f}")
    print(f"     Mass conservation error: {mass_error:.2e}")
    
    if mass_error > 1e-6:
        print("   ⚠ Warning: Mass conservation error is high")
    
except Exception as e:
    print(f"   ✗ LBM iteration failed: {e}")
    sys.exit(1)

try:
    print("\n6. Testing geometry generation...")
    cylinder = create_cylinder(nx=100, ny=50, center=(50, 25), radius=10)
    assert cylinder.shape == (100, 50)
    assert cylinder.dtype == bool
    num_solid = np.sum(cylinder)
    print(f"   ✓ Cylinder created: {num_solid} solid nodes")
except Exception as e:
    print(f"   ✗ Geometry generation failed: {e}")
    sys.exit(1)

try:
    print("\n7. Testing convergence monitoring...")
    monitor = ConvergenceMonitor(window_size=50)
    
    # Add some fake data
    for i in range(100):
        fake_field = np.random.random((100, 50)) * 0.1
        monitor.update(fake_field)
    
    stats = monitor.get_statistics()
    print(f"   ✓ Convergence monitor working")
    print(f"     Tracked {stats['iterations']} iterations")
except Exception as e:
    print(f"   ✗ Convergence monitoring failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All verification tests passed!")
print("="*60)
print("\nNext steps:")
print("1. Open notebooks/main_simulation.ipynb")
print("2. Run the cells sequentially")
print("3. Start with Phase 1 (Poiseuille validation)")
print("\nFor more information, see README.md")
print("="*60)

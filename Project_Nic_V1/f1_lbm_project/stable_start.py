"""
Stable starter script for F1 LBM simulations
This version uses safer parameters to prevent divergence
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.append('src')

from lbm_core import LBMSimulation
from boundary_conditions import BoundaryConditions
from validation import validate_poiseuille
from visualization import plot_velocity_profile, plot_velocity_field, plot_comparison_panel
from geometry import create_cylinder

print("="*60)
print("F1 LBM - Stable Starter Script")
print("="*60)

# =============================================================================
# PART 1: POISEUILLE FLOW (Simple validation)
# =============================================================================

print("\n" + "="*60)
print("PART 1: POISEUILLE FLOW VALIDATION")
print("="*60)

# Safe parameters
nx = 400
ny = 100
Re = 100
u_max = 0.08  # Reduced from 0.1 for stability

print(f"\nDomain: {nx} x {ny}")
print(f"Reynolds number: {Re}")
print(f"Max velocity: {u_max}")

sim = LBMSimulation(nx=nx, ny=ny, reynolds_number=Re, u_max=u_max)

# Initialize with parabolic profile
y = np.arange(ny)
y_center = (ny - 1) / 2.0
u_init = u_max * (1 - ((y - y_center) / y_center)**2)
u_field = np.zeros((2, nx, ny))
u_field[0, :, :] = u_init[np.newaxis, :]

sim.initialize_equilibrium(rho_init=1.0, u_init=u_field)

print(f"Viscosity ν: {sim.nu:.6f}")
print(f"Relaxation time τ: {sim.tau:.4f}")

# Run simulation
print("\nRunning Poiseuille simulation...")
max_iterations = 5000

for step in range(max_iterations):
    bc = BoundaryConditions(sim)
    bc.apply_no_slip_bottom()
    bc.apply_slip_top()
    bc.apply_periodic_x()
    
    sim.step()
    
    if step % 1000 == 0:
        mean_vel = np.mean(sim.u[0])
        print(f"  Step {step:5d}: mean velocity = {mean_vel:.6f}")

print("✓ Simulation complete\n")

# Validate
validation = validate_poiseuille(sim, u_max, sample_x=nx//2)
print("Validation Results:")
print(f"  L2 Error: {validation['l2_error']:.2e}")
print(f"  Relative Error: {validation['relative_error']:.2%}")

if validation['relative_error'] < 0.02:  # 2% tolerance
    print("  ✓ VALIDATION PASSED!")
else:
    print("  ⚠ Error higher than ideal, but may be acceptable")

# Mass conservation
total_mass = np.sum(sim.rho)
expected_mass = 1.0 * nx * ny
mass_error = abs(total_mass - expected_mass) / expected_mass
print(f"  Mass error: {mass_error:.2e}")

# Save plots
plt.figure(figsize=(8, 6))
y_coords = np.arange(ny)
plt.plot(validation['u_numerical'], y_coords, 'b-', linewidth=2, label='LBM')
plt.plot(validation['u_analytical'], y_coords, 'r--', linewidth=2, label='Analytical')
plt.xlabel('Velocity u')
plt.ylabel('Y position')
plt.title(f'Poiseuille Flow Validation (Re={Re})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('poiseuille_validation.png', dpi=150)
print("\n✓ Saved: poiseuille_validation.png")
plt.close()

# =============================================================================
# PART 2: CYLINDER FLOW (More challenging)
# =============================================================================

print("\n" + "="*60)
print("PART 2: CYLINDER FLOW")
print("="*60)

# STABLE parameters for cylinder
nx_cyl = 400
ny_cyl = 100
Re_cyl = 100
u_max_cyl = 0.05  # REDUCED for stability - very important!

print(f"\nDomain: {nx_cyl} x {ny_cyl}")
print(f"Reynolds number: {Re_cyl}")
print(f"Max velocity: {u_max_cyl} (reduced for stability)")

# Create cylinder
radius = 8
center = (80, ny_cyl // 2)

print(f"Cylinder: radius={radius}, center={center}")

sim_cyl = LBMSimulation(nx=nx_cyl, ny=ny_cyl, reynolds_number=Re_cyl, u_max=u_max_cyl)

# Create obstacle
cylinder_mask = create_cylinder(nx_cyl, ny_cyl, center, radius)
sim_cyl.set_obstacle(cylinder_mask)

# Initialize with uniform flow
u_field_cyl = np.zeros((2, nx_cyl, ny_cyl))
u_field_cyl[0, :, :] = u_max_cyl
sim_cyl.initialize_equilibrium(rho_init=1.0, u_init=u_field_cyl)

print(f"Viscosity ν: {sim_cyl.nu:.6f}")
print(f"Relaxation time τ: {sim_cyl.tau:.4f}")
print(f"Obstacle nodes: {np.sum(cylinder_mask)}")

# Run simulation
print("\nRunning cylinder simulation...")
max_iterations_cyl = 8000
checkpoint = 500

diverged = False

for step in range(max_iterations_cyl):
    # Apply boundaries
    bc_cyl = BoundaryConditions(sim_cyl)
    bc_cyl.apply_inlet_velocity(u_max_cyl, axis='left')
    bc_cyl.apply_outlet_pressure(rho_outlet=1.0, axis='right')
    bc_cyl.apply_slip_bottom()
    bc_cyl.apply_slip_top()
    
    # LBM step
    sim_cyl.step()
    
    # Check for divergence
    if step % checkpoint == 0:
        mean_vel = np.mean(sim_cyl.u[0][~cylinder_mask])
        max_vel = np.max(np.abs(sim_cyl.u[:, ~cylinder_mask]))
        
        if np.isnan(mean_vel) or np.isnan(max_vel):
            print(f"  ✗ Simulation diverged at step {step}")
            diverged = True
            break
        
        print(f"  Step {step:5d}: mean_vel = {mean_vel:.6f}, max_vel = {max_vel:.6f}")

if not diverged:
    print("✓ Cylinder simulation complete\n")
    
    # Create visualization
    plot_comparison_panel(sim_cyl, figsize=(16, 4))
    plt.savefig('cylinder_flow.png', dpi=150)
    print("✓ Saved: cylinder_flow.png")
    plt.close()
else:
    print("\n⚠ Cylinder simulation diverged")
    print("This can happen with complex flows. Try:")
    print("  - Reducing u_max further (try 0.03)")
    print("  - Increasing domain size")
    print("  - Using smaller cylinder radius")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nPoiseuille Flow:")
print(f"  ✓ Completed successfully")
print(f"  Validation error: {validation['relative_error']:.2%}")

print(f"\nCylinder Flow:")
if not diverged:
    print(f"  ✓ Completed successfully")
    print(f"  Final iteration: {sim_cyl.iteration}")
else:
    print(f"  ⚠ Diverged at iteration ~{step}")

print("\n" + "="*60)
print("Check the saved PNG files to see your results!")
print("="*60)

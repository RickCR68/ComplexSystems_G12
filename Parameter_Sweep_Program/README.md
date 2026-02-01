# F1 Ground Effect CFD Simulator

A Lattice Boltzmann Method (LBM) implementation for analyzing F1 aerodynamics and ground effect phenomena using computational fluid dynamics.

## Overview

This simulator uses the D2Q9 lattice Boltzmann approach with Smagorinsky turbulence modeling to study airflow around F1 wing geometries near the ground. It includes tools for chaos detection, energy spectrum analysis, and force calculations.

## Core Components

- **lbm_core.py**: D2Q9 LBM solver with Smagorinsky subgrid-scale turbulence model
- **boundaries.py**: Geometric definitions for wind tunnel, ground plane, and F1 wing proxies
- **aerodynamics.py**: Force calculations using momentum exchange method (drag/lift/downforce)
- **analysis.py**: Turbulence analysis tools including energy spectrum computation and visualization
- **Main.ipynb**: Example workflows and parameter sweeps

## Physics Implementation

- Reynolds number-based viscosity calculation
- Equilibrium distribution functions for D2Q9 lattice
- BGK collision operator with dynamic relaxation time
- Bounce-back boundary conditions for solid walls
- Inlet/outlet flow control

## Analysis Features

- Vorticity field visualization
- 2D FFT energy spectrum with radial averaging
- Power law scaling detection (k^-3 Kraichnan, k^-5/3 Kolmogorov)
- Real-time force monitoring for ground effect verification

## Typical Use Case

```python
# Initialize solver
solver = LBMSolver(nx=200, ny=80, reynolds=1000, u_inlet=0.1)

# Define geometry
bounds = TunnelBoundaries(nx=200, ny=80)
bounds.add_ground(type="no_slip")
bounds.add_f1_wing_proxy(x_pos=50, height=5, length=30, slope=0.5)

# Run simulation
for step in range(max_steps):
    solver.collide_and_stream(bounds.mask)
    bounds.apply_inlet_outlet(solver)
    fx, fy = calculate_lift_drag(solver, bounds)
```

## Requirements

- numpy
- matplotlib

## Notes

Force calculations use a restricted domain search for computational efficiency. The Smagorinsky constant (default 0.15) controls subgrid turbulence dissipation and may require tuning for different Reynolds numbers.

# F1 Ground Effect LBM Simulation

**Team 12**: Alfonso Gondro Ostos, Ricardo Carvalho Ribeiro, Nicholas Miot, Xiaoduo Zhuo

## Project Overview

This project investigates the emergent phenomenon of turbulence in Formula 1 ground-effect aerodynamics using the Lattice Boltzmann Method (LBM). We simulate the interaction between airflow and a moving ground plane, focusing on how variations in ride height and airflow viscosity lead to transitions in flow stability and downforce generation.

## Research Questions

1. **Phase Transition Analysis**: At what point and under what conditions does laminar flow transition to turbulent flow? What order of phase transition is it?

2. **Sensitivity Analysis**: How sensitive is turbulent flow to initial conditions such as viscosity, ride height, and pitch angle?

3. **Microscopic to Macroscopic**: How do microscopic particle interactions relate to macroscopic turbulent behavior?

4. **Ground Effect**: How does the inclusion of ground/track boundary conditions affect the transition to turbulent flow?

## Project Structure

```
f1_lbm_project/
├── src/                          # Source code modules
│   ├── lbm_core.py              # Core LBM implementation (D2Q9 lattice)
│   ├── boundary_conditions.py   # Boundary condition implementations
│   ├── geometry.py              # Obstacle geometry generation
│   ├── validation.py            # Analytical validation tools
│   ├── analysis.py              # Turbulence analysis and metrics
│   └── visualization.py         # Plotting and visualization
├── notebooks/                    # Jupyter notebooks
│   └── main_simulation.ipynb    # Main execution notebook
├── tests/                        # Unit tests
│   └── test_lbm.py              # Test suite
├── config/                       # Configuration files
│   └── config.yaml              # Simulation parameters
├── data/                         # Simulation output data
├── visualizations/               # Generated plots and animations
└── README.md                     # This file
```

## Installation & Setup

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- PyYAML
- SciPy
- Jupyter

### Installation

```bash
# Clone or navigate to project directory
cd f1_lbm_project

# Install dependencies
pip install numpy matplotlib pyyaml scipy jupyter --break-system-packages

# Verify installation by running tests
cd tests
python test_lbm.py
```

## Usage

### Quick Start

The main workflow is implemented in a Jupyter notebook:

```bash
cd notebooks
jupyter notebook main_simulation.ipynb
```

### Workflow Phases

#### Phase 1: Minimal Viable Simulation (MVS)
- Validate LBM implementation against Poiseuille flow
- Verify mass and momentum conservation
- Ensure numerical accuracy

#### Phase 2: Cylinder Flow & Reynolds Sweep
- Test with cylinder obstacle
- Observe vortex shedding (Re > 47)
- Validate turbulence detection methods

#### Phase 3: F1 Wing Proxy
- Simulate ground-effect aerodynamics
- Analyze downforce generation
- Compare slip vs no-slip ground boundaries
- Detect laminar-turbulent transition

#### Phase 4: Parameter Sweeps
- Systematically vary Reynolds number
- Test different ride heights
- Map phase transition boundaries

## Key Components

### LBM Core (`lbm_core.py`)

Implements the D2Q9 lattice Boltzmann method with:
- BGK collision operator
- Streaming step
- Bounce-back boundary conditions
- Macroscopic variable calculation

```python
from lbm_core import LBMSimulation

# Create simulation
sim = LBMSimulation(nx=400, ny=100, reynolds_number=500, u_max=0.1)
sim.initialize_equilibrium()

# Run simulation
for step in range(10000):
    # Apply boundaries, then step
    sim.step()
```

### Boundary Conditions (`boundary_conditions.py`)

Supports various boundary types:
- **Slip boundary**: Specular reflection (frictionless wall)
- **No-slip boundary**: Bounce-back (viscous wall)
- **Velocity inlet**: Zou-He velocity BC
- **Pressure outlet**: Zou-He pressure BC
- **Periodic**: For fully-developed flows

### Geometry Generation (`geometry.py`)

Create obstacles:
- Cylinders
- Rectangles
- NACA airfoils
- Simplified F1 wing proxies

### Validation (`validation.py`)

Compare with analytical solutions:
- Poiseuille flow (parabolic velocity profile)
- Lid-driven cavity (Ghia et al. benchmark)
- Mass/momentum conservation checks

### Analysis Tools (`analysis.py`)

Turbulence detection and characterization:
- Turbulent Kinetic Energy (TKE)
- Reynolds stress
- Autocorrelation functions
- Convergence monitoring
- Force calculations (drag, lift coefficients)
- Vortex shedding detection

### Visualization (`visualization.py`)

Generate plots:
- Velocity magnitude fields
- Vorticity contours
- Streamlines
- Convergence histories
- Multi-panel comparisons

## Configuration

Edit `config/config.yaml` to modify simulation parameters:

```yaml
domain:
  nx: 400
  ny: 100

physics:
  reynolds_numbers: [100, 500, 1000, 2000]
  u_max: 0.1

geometry:
  wing:
    chord: 50
    ride_height: 15
    angle_of_attack: 5.0

boundaries:
  bottom_type: 'slip'  # or 'no_slip'
```

## Expected Results

### Poiseuille Flow Validation
- L2 error < 0.01 (1% of u_max)
- Mass conservation to machine precision
- Convergence in ~5000 iterations

### Cylinder Flow (Re = 100)
- Periodic vortex shedding
- Kármán vortex street formation
- Strouhal number St ≈ 0.16-0.17

### F1 Wing Ground Effect
- Increased downforce at lower ride heights
- Transition to turbulence at high Re
- Difference between slip/no-slip ground:
  - No-slip: More boundary layer separation
  - Slip: Higher velocity under wing

## Testing

Run the test suite:

```bash
cd tests
python test_lbm.py
```

Or with pytest:

```bash
pytest test_lbm.py -v
```

Tests cover:
- Lattice initialization
- Mass conservation
- Equilibrium distribution calculation
- Geometry generation
- Poiseuille flow validation

## Performance Notes

### Computational Cost
- Domain 400×100: ~100-200 iterations/second (CPU)
- Domain 800×200: ~25-50 iterations/second (CPU)
- Convergence typically requires 10,000-50,000 iterations

### Optimization Tips
1. Keep u_max < 0.3 for stability (Mach number constraint)
2. Use smaller domains for testing
3. Checkpoint every 500-1000 iterations
4. Consider GPU acceleration for production runs (migrate to Snellius)

### Stability Criteria
- **CFL condition**: u_max × Δt / Δx < 1
- **Viscosity constraint**: τ > 0.5 (τ = 0.5 is inviscid limit)
- **Compressibility**: Keep Mach number Ma = u/c_s < 0.3

## Troubleshooting

### Simulation Diverges (NaN values)
- Reduce u_max (try 0.05 instead of 0.1)
- Check Reynolds number (very high Re may require finer grid)
- Verify boundary conditions are applied correctly

### Poor Convergence
- Increase simulation time (more iterations)
- Check obstacle placement (ensure gap from boundaries)
- Verify viscosity calculation (τ should be > 0.5)

### Mass Not Conserved
- Check boundary condition implementation
- Ensure obstacle mask doesn't change during simulation
- Verify streaming step is correct

## References

1. Krüger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.
2. Pomeau, Y. (2015). "The transition to turbulence: A personal view." *Comptes Rendus Mécanique*.
3. Ghia, U., et al. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*.

## License

This is an academic project for educational purposes.

## Contact

For questions or issues, contact team members:
- Alfonso Gondro Ostos
- Ricardo Carvalho Ribeiro  
- Nicholas Miot
- Xiaoduo Zhuo

---

**Last Updated**: January 2025  
**Project Timeline**: January 19 - January 30, 2025

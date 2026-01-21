# Wind Tunnel LBM Simulation

A Python-based 2D lattice Boltzmann method (LBM) solver for simulating incompressible fluid flow in wind tunnel configurations. Designed for education, research, and validation of fluid dynamics concepts.

## Features

- **2D Lattice Boltzmann Solver (D2Q9)**
  - BGK collision operator
  - Efficient streaming via NumPy
  - Macroscopic property recovery (density, velocity)

- **Flexible Geometry & Boundary Conditions**
  - Rectangular domains with configurable size
  - Circular obstacle placement (easy to extend to other shapes)
  - Inlet, outlet, and no-slip wall boundary conditions
  - Bounce-back collision for solid obstacles

- **Configuration Management**
  - JSON-backed dataclass-based configuration system
  - Pre-defined scenarios (Poiseuille, cylinder flow)
  - Easy parameterization of grid size, viscosity, flow speed, obstacles

- **Results Caching**
  - Automatic caching of simulation results to `/results/` directory
  - Avoid expensive recomputation
  - Reproducible research with saved parameters and fields

- **Visualization & Analysis**
  - Velocity magnitude heatmaps
  - Vorticity field visualization
  - Quiver (arrow) plots for velocity vectors
  - Streamline plots
  - Velocity profile extraction
  - Easy integration with Jupyter notebooks

- **Comprehensive Testing**
  - Unit tests for solver correctness (collision, streaming, macroscopic recovery)
  - Geometry validation tests
  - Configuration loading/saving tests
  - Visualization function tests
  - Target: >60% code coverage on core modules

## Project Structure (simplified)

```
wind-tunnel-sim/
├── simulation/
│   ├── __init__.py
│   ├── core.py            # Unified solver + geometry + BCs
│   ├── cache.py           # JSON caching utilities
│   └── visualization.py   # Plot helpers
├── config/
│   ├── __init__.py
│   ├── parameters.py      # Single SimulationConfig dataclass
│   └── scenarios/
│       ├── poiseuille.json
│       └── cylinder.json
├── tests/
│   ├── test_solver.py     # Core simulation smoke tests
│   ├── test_config.py     # Config load/save tests
│   └── test_visualization.py
├── notebooks/
│   ├── 01_poiseuille_validation.ipynb
│   └── 02_cylinder_flow.ipynb
├── results/
│   └── cached JSON outputs
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone repository** (or extract to working directory)

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
pytest tests/ -v
```

## Quick Start

### 1. Using Jupyter Notebooks (Recommended for exploration)

```bash
jupyter lab
```

Open `notebooks/01_poiseuille_validation.ipynb` to:
- Load configuration
- Run Poiseuille flow validation
- Compare numerical vs. analytical solutions
- Visualize results
- Save to cache

### 2. Running from Python Script

```python
from pathlib import Path
from config.parameters import SimulationConfig
from simulation.core import LBMSimulation
from simulation.cache import SimulationCache, SimulationResult
from simulation.visualization import Visualizer

cfg = SimulationConfig.from_json(Path("config/scenarios/poiseuille.json"))
sim = LBMSimulation(cfg)
cache = SimulationCache("results")

for step in range(cfg.num_iterations):
  sim.step(cfg.inlet_velocity)
  if step % cfg.save_interval == 0:
    print(f"Step {step}/{cfg.num_iterations}")

rho, ux, uy = sim.state.rho, sim.state.ux, sim.state.uy
vort = sim.compute_vorticity()

result = SimulationResult(
  scenario_name=cfg.scenario_name,
  grid_size_x=cfg.grid_size_x,
  grid_size_y=cfg.grid_size_y,
  viscosity=cfg.viscosity,
  inlet_velocity=cfg.inlet_velocity,
  num_iterations=cfg.num_iterations,
  iteration_count=cfg.num_iterations,
  velocity_x=ux.tolist(),
  velocity_y=uy.tolist(),
  vorticity=vort.tolist(),
  density=rho.tolist(),
  reynolds_number=cfg.reynolds_number,
)
cache.save_result(result)

# Plot a quick visualization
Visualizer.plot_velocity_magnitude(ux, uy)
```

## Running the Project

### Automated Testing

Verify the entire project works by running the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=simulation --cov=config --cov-report=html
```

### Quick Smoke Test

Verify core simulation runs without errors:

```bash
python - <<'EOF'
from pathlib import Path
from config.parameters import SimulationConfig
from simulation.core import LBMSimulation

cfg = SimulationConfig.from_json(Path("config/scenarios/poiseuille.json"))
sim = LBMSimulation(cfg)
sim.run(cfg.inlet_velocity, num_steps=100)
print(sim.summary())
print("✓ Simulation core works!")
EOF
```

### Interactive Jupyter Notebooks

For hands-on exploration and visualization:

```bash
jupyter lab
```

Then open either notebook:
- `notebooks/01_poiseuille_validation.ipynb` — Run validation against analytical solution
- `notebooks/02_cylinder_flow.ipynb` — Study flow around a cylinder

### Load Cached Results

After running simulations, view cached outputs:

```python
from simulation.cache import SimulationCache

cache = SimulationCache("results")
result = cache.load_latest("poiseuille")
ux, uy = result.get_velocity_arrays()
vorticity = result.get_vorticity_array()
print(f"Cached at: {result.timestamp}")
```

## Configuration

### Parameters

Configuration is defined in `config/parameters.py` and stored as JSON files in `config/scenarios/`.

**Key Parameters**:
- `grid_size_x`, `grid_size_y`: Domain dimensions (lattice nodes)
- `viscosity`: Kinematic viscosity (lattice units)
- `inlet_velocity`: Bulk inlet velocity (lattice units)
- `num_iterations`: Total time steps to simulate
- `obstacles`: List of circular obstacles (x, y, radius)

**Important Derived Parameters**:
- Reynolds number: `Re = inlet_velocity * grid_size_y / viscosity`
- Relaxation time: `tau = 3 * viscosity + 0.5` (D2Q9 lattice)

### Creating Custom Scenarios

```python
from config.parameters import SimulationConfig, Obstacle
from pathlib import Path

config = SimulationConfig(
    scenario_name="my_scenario",
    grid_size_x=256,
    grid_size_y=128,
    viscosity=0.01,
    inlet_velocity=0.15,
    num_iterations=5000,
    obstacles=[Obstacle(x=80, y=64, radius=10)]
)

# Save for later use
config.to_json(Path("config/scenarios/my_scenario.json"))
```

## Validation Benchmarks

### 1. Poiseuille Flow

**Analytical Solution**: Parabolic velocity profile in channel flow.

The numerical solution is validated against the analytical profile:
$$u_x(y) = u_{max} \cdot 4 \frac{y(H-y)}{H^2}$$

**Acceptance Criteria**:
- Velocity error < 1% at centerline
- Pressure drop matches theoretical prediction
- Fully developed flow after ~5000 iterations

**Run**: `notebooks/01_poiseuille_validation.ipynb`

### 2. Cylinder Flow

**Key Metrics**:
- Drag coefficient ($C_d$): Compare to literature values
- Strouhal number (vortex shedding frequency)
- Vortex wake structure

**Reference Values** (Re ≈ 1280, cylinder diameter D = 16 lattice units):
- $C_d \approx 1.2$ (steady-state)
- Vortex shedding expected around Re > 47

**Run**: `notebooks/02_cylinder_flow.ipynb`

## Results Caching

Simulation results are automatically cached as JSON files in `/results/` to avoid expensive recomputation.

**Cache Contents**:
- `velocity_x`, `velocity_y`: Velocity field arrays
- `vorticity`: Vorticity field
- `density`: Density field
- `drag_coefficient`, `lift_coefficient`: Aerodynamic metrics
- `reynolds_number`: Simulation Reynolds number
- `timestamp`: When result was computed
- `metadata`: Simulation configuration

**Usage**:

```python
from simulation.cache import SimulationCache

cache = SimulationCache("results")

# Save result
cache.save_result(result)

# Load latest result for a scenario
result = cache.load_latest("poiseuille")

# List all cached scenarios
scenarios = cache.list_cached_scenarios()

# Load specific result by path
result = cache.load_by_path(Path("results/poiseuille_20250120_120000.json"))

# Convert cached arrays back to NumPy
ux, uy = result.get_velocity_arrays()
vorticity = result.get_vorticity_array()
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=simulation --cov=config --cov-report=html
```

Coverage target: **>60%** on core simulation and configuration modules.

### Test Organization

- `tests/test_solver.py`: Core D2Q9 solver + BC/geometry mask smoke tests
- `tests/test_config.py`: Parameter loading, saving, validation
- `tests/test_visualization.py`: Plotting and field computation

## Visualization Examples

```python
from simulation.visualization import Visualizer
import numpy as np

# After running simulation, you have ux, uy velocity fields

# Plot velocity magnitude
fig, ax = Visualizer.plot_velocity_magnitude(ux, uy, title="Velocity Magnitude")

# Plot vorticity
vort = Visualizer.compute_vorticity(ux, uy)
fig, ax = Visualizer.plot_vorticity(vort, title="Vorticity Field")

# Quiver plot
fig, ax = Visualizer.plot_velocity_quiver(ux, uy, stride=4)

# Streamlines
fig, ax = Visualizer.plot_streamlines(ux, uy)

# Velocity profile at a given y-location
fig, ax = Visualizer.plot_velocity_profile(ux, y_slice=32)
```

## Common Workflows

### 1. Validate Solver Correctness

```bash
jupyter lab notebooks/01_poiseuille_validation.ipynb
```

Runs Poiseuille flow, checks velocity profile matches analytical solution.

### 2. Study Cylinder Wake

```bash
jupyter lab notebooks/02_cylinder_flow.ipynb
```

Simulates flow around cylinder, computes drag coefficient, visualizes wake.

### 3. Parameter Sweep (Future)

Modify notebook to loop over viscosity/velocity values and study Reynolds number effects.

### 4. Add New Obstacle Geometry

Edit `config/scenarios/custom.json` to define new obstacle positions/sizes, or extend `simulation/core.py` for non-circular obstacles.

## Design Decisions

### 1. Why D2Q9?
- Sufficient for demonstrating LBM concepts
- Low computational cost (~9 fields per node)
- Well-validated for incompressible flows
- Easy to understand and teach

### 2. Why NumPy?
- Vectorized operations for efficiency
- No external dependencies beyond standard scientific stack
- Clear, readable code for educational purposes
- Easy path to GPU acceleration via CuPy (future)

### 3. Why JSON for Configuration?
- Human-readable and editable
- No external dependencies
- Easy to version control
- Simple to extend with new parameters

### 4. Why Results Caching?
- Expensive simulations (10k+ iterations) take time
- Decouples simulation from analysis
- Enables reproducible research
- Allows re-analysis without re-computation

## Performance Notes

**Typical Runtimes** (Intel Core i7, 1 CPU thread):
- Poiseuille (256×64, 5000 steps): ~30 seconds
- Cylinder (256×128, 10000 steps): ~2-3 minutes

**Memory Usage**:
- Solver: ~7 MB per million grid points (9 distribution functions × float32)
- 256×128 domain: ~1-2 MB

## Troubleshooting

### Simulation Diverges (NaN/Inf values)

**Cause**: Relaxation time τ < 0.5 or Mach number too high

**Solution**:
- Increase `viscosity` (raises τ)
- Decrease `inlet_velocity` (lowers Mach)
- Check `config.relaxation_time` property

### Slow Performance

**Solutions**:
- Reduce grid size (`grid_size_x`, `grid_size_y`)
- Fewer iterations (`num_iterations`)
- (Future) Use GPU acceleration

### Import Errors

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

## Future Extensions

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for planned enhancements:
- 3D solver (D3Q19)
- Turbulence modeling
- GPU acceleration
- Thermal effects
- Advanced boundary conditions
- Immersed boundary method

## References

### Lattice Boltzmann Theory
- Succi, S. (2001). "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond."
- Krüger, T., et al. (2017). "The Lattice Boltzmann Method: Principles and Practice."

### LBMpy Library
- [lbmpy Documentation](https://lbmpy.readthedocs.io/)
- GitHub: https://github.com/walberla/lbmpy

### Validation Benchmarks
- Ghia, U., et al. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations."
- Williamson, C.H.K. (1996). "Vortex Dynamics in the Cylinder Wake."

## Contributing

To add new features:
1. Create feature branch
2. Implement with comprehensive docstrings
3. Add unit tests (aim for >60% coverage)
4. Update documentation
5. Validate against benchmark cases
6. Submit pull request

## License

[Specify your license, e.g., MIT, Apache 2.0]

## Contact

[Team Lead Name] - [email]

---

**Last Updated**: January 20, 2026  
**Status**: Active Development

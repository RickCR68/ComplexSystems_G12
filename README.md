# F1 Ground Effect LBM Simulation

A Lattice Boltzmann Method (LBM) simulation framework for studying aerodynamic ground effect in Formula 1 cars.

## Features

- **D2Q9 LBM Solver** with BGK collision operator
- **Smagorinsky turbulence model** for high Reynolds number flows
- **Multiple geometries**: Simple shapes (triangle, cylinder) to full F1 car profiles
- **Configurable ride height** for ground effect studies
- **Parameter sweep** functionality for systematic exploration
- **Automatic result saving** for reproducibility
- **Publication-quality visualizations**

## Project Structure

```
f1_lbm_simulation/
├── main.ipynb          # Main notebook for running experiments
├── lbm_core.py         # Core LBM solver
├── geometry.py         # Obstacle geometries and boundaries
├── aerodynamics.py     # Force calculations (Momentum Exchange Method)
├── analysis.py         # Flow diagnostics and turbulence statistics
├── visualization.py    # Plotting and animation tools
├── experiment.py       # Configuration and result management
├── runner.py           # High-level simulation interface
├── __init__.py         # Package exports
├── results/            # Saved experiment results
├── figures/            # Generated figures
└── configs/            # Saved configurations
```

## Quick Start

### 1. Open the main notebook

```bash
jupyter notebook main.ipynb
```

### 2. Run a quick simulation

```python
from runner import quick_run

results = quick_run(
    reynolds=2000,
    ride_height=5,
    geometry='f1_wing_simple',
    steps=5000
)
```

### 3. Run a parameter sweep

```python
from experiment import SimulationConfig
from runner import run_parameter_sweep

config = SimulationConfig(
    geometry_type='f1_wing_simple',
    total_steps=5000
)

sweep_params = {
    'ride_height': [3, 5, 7, 10, 15],
    'reynolds': [500, 1000, 2000]
}

sweep, results = run_parameter_sweep(config, sweep_params)

# Analyze results
df = sweep.get_results_dataframe()
print(df)
```

## Available Geometries

| Type | Description | Use Case |
|------|-------------|----------|
| `triangle` | Simple triangular obstacle | Validation |
| `rectangle` | Bluff body rectangle | Drag studies |
| `cylinder` | Circular cylinder | Canonical validation |
| `naca_airfoil` | NACA 4-digit airfoil | Wing sections |
| `f1_wing_simple` | Simplified front wing | Ground effect |
| `f1_car_full` | Complete F1 side profile | Full car analysis |

## Key Parameters

### Physics
- `reynolds`: Reynolds number (typically 100-10000)
- `u_inlet`: Inlet velocity in lattice units (keep < 0.15 for stability)
- `cs_smag`: Smagorinsky constant (0.1-0.2, set to 0 to disable)

### Geometry
- `ride_height`: Ground clearance in lattice units
- `geometry_scale`: Scaling factor for obstacle size
- `ground_type`: `"no_slip"` (stationary) or `"moving"` (realistic)

### Simulation
- `total_steps`: Number of simulation steps
- `output_interval`: Steps between force recordings
- `convergence_tolerance`: Relative tolerance for convergence check

## Output Files

Each experiment creates a directory with:

| File | Description |
|------|-------------|
| `config.json` | Full configuration (for reproducibility) |
| `summary.json` | Key results in JSON format |
| `summary.txt` | Human-readable summary |
| `force_history.csv` | Time series of drag/lift |
| `final_state.npz` | Flow field snapshots (velocity, density) |
| `figures/` | Generated visualizations |

## Physics Background

### Lattice Boltzmann Method

The LBM solves the Boltzmann equation on a discrete lattice. The D2Q9 scheme uses 9 velocity directions in 2D:

```
6   2   5
  \ | /
3 - 0 - 1
  / | \
7   4   8
```

### BGK Collision

The collision step relaxes distributions toward equilibrium:
```
f_i(t+1) = f_i(t) - (f_i - f_eq) / tau
```

### Smagorinsky Model

For high Reynolds numbers, the Smagorinsky subgrid model adds eddy viscosity:
```
nu_eff = nu_0 + (C_s * Delta)^2 * |S|
```

### Momentum Exchange Method

Forces are computed by summing momentum transferred during bounce-back:
```
F = sum_boundary(2 * f_incoming * c)
```

## Tips for Best Results

1. **Stability**: Keep `u_inlet < 0.15` and ensure `tau > 0.51`
2. **Resolution**: Higher `ny` gives better accuracy but slower simulation
3. **Convergence**: Monitor forces - they should stabilize
4. **Ground Effect**: Lower ride height = more downforce (to a point!)
5. **Turbulence**: Enable Smagorinsky model (`cs_smag > 0`) for high Re

## Example Results

### Ground Effect Study

Lower ride height increases downforce due to accelerated flow under the car (Venturi effect). However, very low ride heights can cause flow separation.

```
Ride Height | Downforce | Drag
------------|-----------|-----
15          | 0.012     | 0.045
10          | 0.025     | 0.048
5           | 0.058     | 0.052
3           | 0.082     | 0.061
```

## Dependencies

- NumPy
- Matplotlib
- (Optional) Pandas - for DataFrame analysis
- (Optional) Pillow - for GIF animations

## References

1. Krüger et al., "The Lattice Boltzmann Method" (2017)
2. Pope, "Turbulent Flows" (2000)
3. Mei et al., "Force evaluation in the LBM" (2002)

## License

MIT License - See LICENSE file for details.

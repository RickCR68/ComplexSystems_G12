# F1 Turbulence Simulation - Experiment Framework

A modular framework for running F1 ground-effect aerodynamics simulations using Lattice Boltzmann Method (LBM) with automated parameter sweeps and result management.

## Features

- 🔬 **Single Experiment Mode**: Run individual simulations with custom parameters
- 🔄 **Automated Parameter Sweeps**: Systematically explore Reynolds number and ride height
- 💾 **Automatic Result Saving**: All results saved with metadata and configuration
- 📊 **Built-in Analysis**: Automatic generation of comparison plots and summary statistics
- 🎯 **Modular Design**: Clean separation between configuration, execution, and analysis

## Quick Start

### Installation

```bash
# Install required packages
pip install numpy matplotlib pandas tqdm

# Or use the requirements file
pip install -r requirements.txt
```

### Run Your First Simulation

**Option 1: Python Script**
```bash
python main.py                    # Run single experiment
python main.py --quick-test       # Quick test run
python main.py --sweep reynolds   # Sweep Reynolds numbers
python main.py --sweep full       # Full parameter sweep
```

**Option 2: Jupyter Notebook**
```bash
jupyter notebook main.ipynb
```

Then follow the notebook cells for step-by-step execution.

## Usage Examples

### 1. Single Experiment

```python
from config import SimulationConfig
from runner import run_and_save

# Configure your experiment
config = SimulationConfig(
    reynolds=12000,      # Reynolds number
    ride_height=7,       # Distance from ground
    total_steps=10000,
    output_dir="results/my_experiment"
)

# Run and save
output_path = run_and_save(config, verbose=True)
```

### 2. Reynolds Number Sweep

```python
from parameter_sweep import run_parameter_sweep

# Sweep over Reynolds numbers
sweep = run_parameter_sweep(
    reynolds_values=[5000, 10000, 15000, 20000],
    ride_height_values=[5],  # Fixed height
    output_dir="results/reynolds_sweep"
)
```

### 3. Ride Height Sweep

```python
# Sweep over ride heights
sweep = run_parameter_sweep(
    reynolds_values=[10000],  # Fixed Reynolds
    ride_height_values=[3, 5, 7, 10, 15],
    output_dir="results/height_sweep"
)
```

### 4. Full Parameter Sweep

```python
# Sweep both parameters (full factorial)
sweep = run_parameter_sweep(
    reynolds_values=[5000, 10000, 15000],
    ride_height_values=[3, 5, 7, 10],
    output_dir="results/full_sweep"
)
# This runs 3 × 4 = 12 simulations
```

### 5. Custom Configuration

```python
from config import SimulationConfig
from parameter_sweep import ParameterSweep

# Create custom configurations
configs = [
    SimulationConfig(
        reynolds=8000,
        ride_height=4,
        wing_type="triangle",
        run_id="config_1"
    ),
    SimulationConfig(
        reynolds=12000,
        ride_height=6,
        wing_type="reverse_triangle",
        ground_type="slip",
        run_id="config_2"
    ),
]

# Run custom sweep
sweep = ParameterSweep(configs, output_dir="results/custom")
sweep.run()
```

## Configuration Parameters

### Physics Parameters
- `reynolds`: Reynolds number (controls turbulence intensity)
  - Range: 1,000 - 50,000
  - Typical: 10,000 - 20,000
- `u_inlet`: Inlet velocity (default: 0.1)
- `cs_smag`: Smagorinsky constant for turbulence model (default: 0.15)

### Geometry Parameters
- `ride_height`: Distance from ground to wing base
  - Range: 3 - 20 grid units
  - Lower = stronger ground effect
- `wing_x_pos`: Horizontal position of wing (default: 50)
- `wing_length`: Wing length (default: 30)
- `wing_slope`: Wing angle (default: 0.5)
- `wing_type`: "triangle" or "reverse_triangle"
- `ground_type`: "no_slip" (friction) or "slip" (no friction)

### Domain Parameters
- `nx`: Grid points in x-direction (default: 400)
- `ny`: Grid points in y-direction (default: 100)
- Higher resolution = better accuracy but slower

### Simulation Control
- `total_steps`: Number of time steps (default: 10,000)
- `monitor_interval`: Steps between force measurements (default: 100)
- `dashboard_interval`: Steps between complexity plots (default: 500)

### Output Control
- `save_flow_field`: Save final velocity field (default: True)
- `save_force_history`: Save force time series (default: True)
- `save_dashboard`: Save complexity dashboard plots (default: False)
- `output_dir`: Base directory for results (default: "results")

## Output Structure

Each simulation creates a directory with:

```
results/
├── single_runs/
│   └── Re10000_h5/
│       ├── config.json           # Configuration used
│       ├── force_history.npz     # Time series data
│       ├── force_history.png     # Force plot
│       ├── final_flow.png        # Flow field visualization
│       ├── final_flow_field.npz  # Raw flow data
│       └── summary_stats.json    # Summary statistics
│
└── parameter_sweep/
    ├── sweep_metadata.json       # Sweep information
    ├── sweep_summary.csv         # All results in table
    ├── parameter_heatmap.png     # 2D comparison plot
    ├── reynolds_comparison.png   # Reynolds sweep plot
    ├── height_comparison.png     # Height sweep plot
    ├── force_comparison.png      # Drag vs Lift scatter
    └── Re10000_h5/...            # Individual run results
```

## Loading and Analyzing Results

### Load Sweep Summary

```python
import pandas as pd

# Load all results from a sweep
df = pd.read_csv("results/full_sweep/sweep_summary.csv")

# Display summary
print(df.describe())
print(df.head())

# Filter results
best_downforce = df[df['downforce_ratio'] > 0.8]
```

### Load Individual Run

```python
import numpy as np

# Load force history
data = np.load("results/single_runs/Re10000_h5/force_history.npz")
steps = data['steps']
drag = data['drag']
lift = data['lift']

# Load configuration
import json
with open("results/single_runs/Re10000_h5/config.json") as f:
    config = json.load(f)
```

### Custom Analysis

```python
import matplotlib.pyplot as plt

# Plot drag vs Reynolds for different heights
for height in df['ride_height'].unique():
    subset = df[df['ride_height'] == height]
    plt.plot(subset['reynolds'], subset['mean_drag'], 
             marker='o', label=f'Height {height}')

plt.xlabel("Reynolds Number")
plt.ylabel("Mean Drag")
plt.legend()
plt.show()
```

## Performance Guidelines

### Computational Cost (approximate)

| Configuration | Grid Size | Steps | Time |
|--------------|-----------|-------|------|
| Quick Test | 200×50 | 1,000 | ~30s |
| Standard | 400×100 | 10,000 | ~5 min |
| High Resolution | 800×200 | 15,000 | ~30 min |

### Parameter Sweep Cost

- Single parameter sweep (4 values): ~20-25 minutes
- Full sweep (3×4 = 12 configs): ~60 minutes
- Tip: Use `--quick-test` first to verify your setup

## Module Structure

```
.
├── config.py              # Configuration management
├── runner.py              # Single simulation execution
├── parameter_sweep.py     # Batch experiment management
├── main.py                # Command-line interface
├── main.ipynb             # Jupyter notebook interface
├── lbm_core.py           # LBM solver (original)
├── boundaries.py          # Geometry setup (original)
├── aerodynamics.py        # Force calculation (original)
├── analysis.py            # Complexity analysis (original)
└── video_maker.py         # Animation generator (original)
```

## Advanced Features

### Predefined Configurations

```python
from config import QUICK_TEST, STANDARD_RUN, HIGH_RESOLUTION

# Use predefined configs
output = run_and_save(QUICK_TEST)
```

### Progress Tracking

All parameter sweeps include progress bars via `tqdm`:
```
Running sweep: 100%|████████| 12/12 [1:02:45<00:00, 313.77s/it]
```

### Error Handling

Parameter sweeps continue even if individual runs fail:
```python
sweep.run(continue_on_error=True)  # Default behavior
```

Failed runs are recorded in `sweep_metadata.json`.

## Troubleshooting

### Issue: Simulation is too slow
**Solution**: Reduce resolution or steps
```python
config = SimulationConfig(
    nx=200,  # Half resolution
    ny=50,
    total_steps=5000  # Half steps
)
```

### Issue: Not enough downforce
**Solution**: Reduce ride height or increase Reynolds number
```python
config = SimulationConfig(
    ride_height=3,  # Lower to ground
    reynolds=15000   # Higher turbulence
)
```

### Issue: Memory error
**Solution**: Reduce domain size
```python
config = SimulationConfig(
    nx=300,
    ny=75
)
```

## Citation

If you use this code for research, please cite:
- Dapena-García et al. (preprint): Lattice Boltzmann Method for turbulence modeling

## License

This code is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository.

---

**Happy Simulating! 🏎️💨**

# Quick Start Guide - F1 Turbulence Simulation

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip install numpy matplotlib pandas tqdm

# 2. Verify setup
python test_setup.py
```

Expected output: "✓ ALL TESTS PASSED - SYSTEM READY FOR EXPERIMENTS"

---

## Your First Experiment (10 minutes)

### Option A: Command Line

```bash
# Quick test (30 seconds)
python main.py --quick-test

# Standard experiment (5 minutes)
python main.py
```

### Option B: Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

Then run the "Setup" cell and choose any experiment section.

---

## Common Workflows

### 1. Test a Single Configuration

**Edit** `main.py` line 24-50 to set your parameters:
```python
config = SimulationConfig(
    reynolds=12000,      # ← Your Reynolds number
    ride_height=7,       # ← Your ride height
    total_steps=10000
)
```

**Run:**
```bash
python main.py
```

**Results:** Check `results/single_runs/Re12000_h7/`

---

### 2. Sweep Reynolds Number

```bash
python main.py --sweep reynolds
```

Tests: Re = 5000, 10000, 15000, 20000 at fixed height

**Time:** ~20 minutes

**Results:** Check `results/reynolds_sweep/sweep_summary.csv`

---

### 3. Sweep Ride Height

```bash
python main.py --sweep height
```

Tests: Height = 3, 5, 7, 10, 15 at fixed Reynolds

**Time:** ~25 minutes

**Results:** Check `results/height_sweep/sweep_summary.csv`

---

### 4. Full Parameter Sweep

```bash
python main.py --sweep full
```

Tests: 3 Reynolds × 4 Heights = 12 simulations

**Time:** ~60 minutes

**Results:** Check `results/full_sweep/` for:
- `sweep_summary.csv` - All results table
- `parameter_heatmap.png` - 2D visualization
- Individual run folders with detailed data

---

## Analyze Results

### Load Sweep Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/full_sweep/sweep_summary.csv")

# Show summary
print(df.describe())

# Find best downforce configuration
best = df.nsmallest(1, 'mean_lift')  # Most negative lift
print(f"Best: Re={best['reynolds'].values[0]}, h={best['ride_height'].values[0]}")

# Plot
plt.figure(figsize=(10, 6))
for height in df['ride_height'].unique():
    subset = df[df['ride_height'] == height]
    plt.plot(subset['reynolds'], subset['mean_lift'], 
             marker='o', label=f'Height {height}')
plt.xlabel("Reynolds Number")
plt.ylabel("Lift (Negative = Downforce)")
plt.legend()
plt.show()
```

---

## Customize Your Sweep

**Edit** `main.py` line 137-155:

```python
def run_full_parameter_sweep():
    reynolds_values = [8000, 12000, 16000]     # ← Your values
    ride_height_values = [4, 6, 8, 10, 12]    # ← Your values
    
    base_config = SimulationConfig(
        nx=400,              # ← Grid resolution
        ny=100,
        total_steps=10000,   # ← Simulation length
        wing_type="triangle" # ← Geometry type
    )
```

**Run:**
```bash
python main.py --sweep full
```

---

## Understanding Results

### Output Files

Each run creates:
```
Re10000_h5/
├── config.json           # What parameters were used
├── force_history.npz     # Time series: drag(t), lift(t)
├── force_history.png     # Quick visualization
├── final_flow.png        # Flow field snapshot
└── summary_stats.json    # Mean, std, downforce ratio
```

### Key Metrics

- **mean_drag**: Average drag force (always positive)
- **mean_lift**: Average lift force (negative = downforce)
- **downforce_ratio**: Fraction of time with lift < 0
  - 1.0 = Always downforce (good!)
  - 0.5 = Half the time
  - 0.0 = Never downforce
- **std_lift/drag**: Force fluctuation (higher = more turbulent)

---

## Troubleshooting

### "Simulation too slow"
→ Reduce resolution or steps:
```python
config = SimulationConfig(
    nx=200,           # Half resolution
    ny=50,
    total_steps=5000  # Half steps
)
```

### "Not enough downforce"
→ Lower ride height:
```python
config = SimulationConfig(
    ride_height=3     # Closer to ground
)
```

### "Results look weird"
→ Run verification test:
```bash
python test_setup.py
```

---

## Next Steps

1. ✅ Run verification: `python test_setup.py`
2. ✅ Run quick test: `python main.py --quick-test`
3. ✅ Run your first sweep: `python main.py --sweep reynolds`
4. ✅ Analyze results using notebook
5. ✅ Customize parameters for your research

**Full documentation:** See `README.md`

**Questions?** Check the troubleshooting section in README.md

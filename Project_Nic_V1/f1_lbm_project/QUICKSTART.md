# Quick Start Guide - F1 LBM Project

## Getting Started in 5 Minutes

### Step 1: Verify Installation
```bash
cd f1_lbm_project
python verify_installation.py
```

You should see: "✓ All verification tests passed!"

### Step 2: Open the Main Notebook
```bash
cd notebooks
jupyter notebook main_simulation.ipynb
```

### Step 3: Run Phase 1 - Poiseuille Validation

In the notebook, run cells sequentially through Phase 1:

1. **Setup and Imports** - Load all modules
2. **Load Configuration** - Read parameters from YAML
3. **Setup Poiseuille Flow** - Initialize channel flow
4. **Run Simulation** - Execute ~5000 iterations
5. **Validate** - Compare with analytical solution
6. **Visualize** - See velocity profiles and convergence

**Expected Results:**
- L2 error < 0.01
- Mass conservation error < 1e-10
- Parabolic velocity profile matching theory

### Step 4: Run Phase 2 - Cylinder Flow

Continue in the notebook through Phase 2:

1. **Setup Cylinder** - Create obstacle at Re=100
2. **Run Simulation** - Execute ~10000 iterations
3. **Visualize** - See vortex shedding and Kármán street

**Expected Results:**
- Periodic vortex shedding behind cylinder
- Variance increases (transition detection)
- Beautiful vorticity patterns

### Step 5: Run Phase 3 - F1 Wing

Continue to Phase 3 for the main project:

1. **Setup Wing** - Create F1 wing proxy with ground
2. **Run Simulation** - Execute ~15000 iterations
3. **Analyze** - Calculate forces and turbulence metrics
4. **Visualize** - See ground effect and flow patterns

**Expected Results:**
- Downforce generation (negative lift)
- Flow acceleration under wing
- Potential turbulent transition at high Re

## Quick Commands Reference

### Run Tests
```bash
cd tests
python test_lbm.py
```

### Modify Parameters
Edit `config/config.yaml`:
```yaml
physics:
  reynolds_numbers: [100, 500, 1000]  # Try different Re
  
geometry:
  wing:
    ride_height: 15  # Change ride height
    angle_of_attack: 5.0  # Adjust angle
    
boundaries:
  bottom_type: 'slip'  # Try 'no_slip'
```

### Save Your Work
Visualizations automatically save to `visualizations/`
Data snapshots save to `data/`

## Troubleshooting

**Problem**: Simulation diverges (NaN values)
- **Solution**: Reduce `u_max` in config.yaml to 0.05

**Problem**: Too slow
- **Solution**: Reduce domain size or max_iterations in config.yaml

**Problem**: Can't see convergence
- **Solution**: Increase max_iterations to 20000-50000

## Tips for Success

1. **Start Small**: Use small domains (400×100) for testing
2. **Checkpoint Often**: Save data every 500 iterations
3. **Watch Convergence**: Monitor variance to detect transitions
4. **Compare Boundaries**: Run same setup with slip vs no-slip ground
5. **Document Everything**: The notebook includes explanatory text

## Next-Level: Parameter Sweeps

Once comfortable, try systematic sweeps:

```python
Re_values = [100, 500, 1000, 2000]
ride_heights = [10, 15, 20, 25]

for Re in Re_values:
    for h in ride_heights:
        # Run simulation
        # Save results
        # Create comparison plots
```

## File Structure Quick Reference

```
f1_lbm_project/
├── notebooks/main_simulation.ipynb  ← START HERE
├── src/                             ← Python modules
├── config/config.yaml               ← Edit parameters
├── visualizations/                  ← Output plots
├── data/                            ← Output data
└── README.md                        ← Full documentation
```

## Success Checklist

- [ ] Verification script passes
- [ ] Poiseuille flow validates (error < 1%)
- [ ] Cylinder shows vortex shedding
- [ ] Wing simulation completes
- [ ] Visualizations look good
- [ ] Ready for parameter sweeps!

## Getting Help

1. Read error messages carefully
2. Check the main README.md
3. Review test_lbm.py for examples
4. Consult the inline notebook documentation

Good luck with your simulations! 🏎️💨

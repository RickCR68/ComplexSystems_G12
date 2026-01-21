# Quick Reference: Turbulence Modeling Features

## 📋 Cheat Sheet

### Enable/Disable Features
```python
sim.use_turbulence_model = True        # On by default
sim.cs = 0.15                          # Smagorinsky constant (0.1-0.2 range)
```

### Check Stability
```python
# During simulation
print(f"Velocity: {sim.diagnostics['max_velocity'][-1]:.2f}")
print(f"Mach: {sim.diagnostics['max_mach'][-1]:.3f}")
print(f"ν_t: {sim.diagnostics['mean_viscosity'][-1]:.6f}")

# After simulation
if sim.diagnostics['instability_detected']:
    print("Instability was detected")
```

### Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| Velocity explodes | ↓ inlet_velocity or ↑ viscosity |
| Mach too high | ↓ inlet_velocity |
| ν_t = 0 (inactive) | Add obstacles to create gradients |
| Simulation crashes | Catch RuntimeError or check Mach |

### Parameter Tuning

```python
# For stability-focused studies
config.viscosity = 0.05
sim.cs = 0.20
config.inlet_velocity = 0.01

# For turbulence-focused studies
config.viscosity = 0.01
sim.cs = 0.15
config.inlet_velocity = 0.03
# (Plus: increase grid resolution)
```

### Plot Results

```python
import matplotlib.pyplot as plt

plt.subplot(2, 2, 1)
plt.plot(sim.diagnostics['max_velocity'])
plt.ylabel('Max Velocity'); plt.title('Stability')

plt.subplot(2, 2, 2)
plt.plot(sim.diagnostics['mean_viscosity'])
plt.ylabel('ν_t'); plt.title('Turbulence Model Activity')

plt.subplot(2, 2, 3)
plt.plot(sim.diagnostics['max_mach'])
plt.ylabel('Mach'); plt.title('Compressibility')

plt.subplot(2, 2, 4)
plt.plot(sim.diagnostics['divergence_norm'])
plt.ylabel('|∇·u|'); plt.title('Divergence Check')

plt.tight_layout()
```

## 🔬 Key Equations

**Eddy Viscosity:**
$$\nu_t = (C_s \Delta)^2 |S|$$

**Strain Rate Magnitude:**
$$|S| = \sqrt{2(S_{xx}^2 + S_{yy}^2 + 2S_{xy}^2)}$$

**Effective Relaxation Time:**
$$\tau_{eff} = \tau_0 + 3\nu_t$$

**Mach Number (check incompressibility):**
$$Ma = \frac{u}{c_s} \quad \text{where} \quad c_s = \frac{1}{\sqrt{3}}$$

## 📊 Performance Guide

| Scenario | Stable? | Max Steps | Recommendation |
|----------|---------|-----------|-----------------|
| Poiseuille (no obstacles) | ✅ Yes | 1000+ | Use as baseline |
| Circle obstacle (r=2) | ⚠️ Marginal | 50-100 | Increase resolution |
| Triangle (size 1×3) | ⚠️ Marginal | 50-100 | Same |
| Any obstacle (128×32 grid) | ⚠️ Marginal | <100 | Increase to 256×64+ |

## 🎯 Research Workflow

1. **Baseline** → Run Poiseuille, verify ν_t behavior
2. **Validate** → Check diagnostics make sense
3. **Experiment** → Vary Cs, inlet_velocity, viscosity
4. **Measure** → Compare diagnostics across cases
5. **Report** → Include diagnostics plots in paper

## 📖 Full Documentation

- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** ← Start here
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** ← Detailed examples
- **[TURBULENCE_IMPROVEMENTS.md](TURBULENCE_IMPROVEMENTS.md)** ← Physics & formulas

## 🚀 Next Steps

1. Run a few simulations with different `cs` values
2. Plot the 4 diagnostic metrics
3. Try Poiseuille vs. Poiseuille + small obstacle
4. Decide if you want to implement halfway bounce-back

Good luck! 🌪️

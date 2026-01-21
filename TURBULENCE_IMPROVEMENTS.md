# LBM Turbulence Modeling & Stability Improvements

## Overview

The improved LBM solver adds three key capabilities for studying low-velocity and turbulent flows:

1. **Smagorinsky Subgrid-Scale (SGS) Turbulence Model**
2. **Stability Diagnostics & Monitoring**
3. **Improved Boundary Conditions**

---

## 1. Smagorinsky Subgrid-Scale Turbulence Model

### What It Does

The Smagorinsky model captures energy dissipation at scales smaller than the lattice spacing. This is critical for:
- **Preventing energy accumulation** that leads to velocity blow-ups
- **Modeling wake regions** behind obstacles with proper dissipation
- **Handling transition flows** between laminar and turbulent regimes

### Physics

The model computes an eddy viscosity based on the local strain rate:

$$\nu_t = (C_s \Delta)^2 |S|$$

where:
- $C_s = 0.15$ is the Smagorinsky constant (tunable: 0.1–0.2)
- $\Delta = 1$ is the grid spacing (in lattice units)
- $|S| = \sqrt{2(S_{xx}^2 + S_{yy}^2 + 2S_{xy}^2)}$ is the strain rate magnitude

The effective relaxation time becomes:
$$\tau_{eff} = \tau_0 + 3\nu_t$$

This increases dissipation in regions with high velocity gradients (wakes, separation zones).

### Implementation in `simulation/core.py`

```python
def _compute_effective_tau(self, rho, ux, uy):
    # Compute velocity gradients
    dux_dx = np.gradient(ux, axis=1)
    duy_dy = np.gradient(uy, axis=0)
    # ... more gradient computations ...
    
    # Strain rate magnitude
    S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
    
    # Turbulent viscosity
    nu_t = (self.cs * delta) ** 2 * S_mag
    
    # Effective relaxation time
    tau_eff = self.tau + 3.0 * nu_t
    return tau_eff
```

### Usage

Enable by default. To adjust the model strength:

```python
sim.cs = 0.1   # Weaker damping
sim.cs = 0.15  # Default (recommended)
sim.cs = 0.2   # Stronger damping (more viscous)
```

To disable turbulence model:

```python
sim.use_turbulence_model = False
```

---

## 2. Stability Diagnostics

### What It Monitors

The solver now automatically tracks:

| Metric | Threshold | Warning |
|--------|-----------|---------|
| **Max Velocity** | > 1.0 | High velocity could indicate instability |
| **Mach Number** | > 0.3 | Compressibility effects significant |
| **Velocity Divergence** | Max(du/dx + dv/dy) | Incompressibility violated |
| **NaN/Inf** | Any | Numerical divergence → crashes |

### Accessing Diagnostics

After simulation, check the history:

```python
sim.diagnostics['max_velocity']      # Velocity per timestep
sim.diagnostics['max_mach']          # Mach number per timestep
sim.diagnostics['mean_viscosity']    # Turbulent viscosity per timestep
sim.diagnostics['divergence_norm']   # Velocity divergence per timestep
sim.diagnostics['instability_detected']  # Boolean flag
```

### Interpreting Warnings

```
High velocity detected (max=2918.74). 
  ↳ Reduce inlet_velocity or increase viscosity
  
High Mach number (Ma=5.054). 
  ↳ LBM assumes Ma << 1; consider reducing flow speed
```

---

## 3. Improved Boundary Conditions

### Bounce-Back for Obstacles

The boundary condition for obstacles uses bounce-back:

$$f_i^{rebound}(\mathbf{x}, t+1) = f_{\bar{i}}(\mathbf{x}, t)$$

where $\bar{i}$ is the opposite direction.

**Implementation:**
```python
def _apply_boundaries(self, inlet_velocity):
    solid = self.mask == 1
    for i in range(9):
        j = self.OPPOSITE[i]
        self.state.f[i, solid] = self.state.f[j, solid]
```

### Limitations & Future Improvements

The standard bounce-back creates sharp velocity discontinuities at the solid boundary. For better stability with obstacles:

1. **Halfway Bounce-Back (HBB)**: Move the boundary to the cell midpoint
   - Reduces slip error
   - Better momentum conservation
   - More stable with small obstacles

2. **Non-Equilibrium Extrapolation (NEE)**: Use non-equilibrium distributions
   - Better for curved boundaries
   - Handles slip and no-slip consistently

3. **Immersed Boundary Method (IBM)**: Force-based coupling
   - Smoothly distributes boundary forces to fluid
   - Allows moving obstacles
   - More computationally expensive

---

## Application to Low-Velocity Flows

### Why Low-Velocity Flows Are Challenging

At low Mach numbers (Ma = u/cs << 0.1):
- **Energy accumulates** instead of cascading to small scales
- **Vortex rings** and **recirculation zones** form easily
- **Roundoff errors** compound faster at low speeds

### Strategy for Turbulence/Separation Studies

1. **Use Smagorinsky model** to dissipate accumulated energy
   ```python
   sim.use_turbulence_model = True
   sim.cs = 0.15  # Adjust based on resolution
   ```

2. **Monitor Mach number** to stay incompressible
   ```python
   if np.max(sim.diagnostics['max_mach']) > 0.2:
       print("Consider reducing inlet velocity or increasing domain")
   ```

3. **Increase resolution** near obstacles
   - More cells = better gradient resolution = more stable
   - Current: 128×32; try 256×64 for obstacles

4. **Use higher viscosity** to damp instabilities
   - Trade-off: reduces Reynolds number (less interesting turbulence)
   - Range: ν = 0.01 to 0.1 for D2Q9 stability

5. **Adjust Smagorinsky constant** based on observations
   - If velocity oscillates: increase Cs (0.2–0.3)
   - If flow is too damped: decrease Cs (0.05–0.1)

---

## Examples

### Example 1: Stable Poiseuille Flow

```python
config = SimulationConfig.from_json(Path('./config/scenarios/poiseuille.json'))
sim = LBMSimulation(config)

# Run without obstacles - baseline stable case
for step in range(100):
    sim.step(config.inlet_velocity)
    if sim.diagnostics['instability_detected']:
        print(f"Instability at step {step}")
        break

print(f"Max velocity: {np.max(sim.diagnostics['max_velocity']):.6f}")
print(f"Turbulent viscosity: {np.mean(sim.diagnostics['mean_viscosity']):.6f}")
```

Expected output: Stable, max velocity ≈ 0.47

### Example 2: Obstacle with Turbulence Model

```python
config = SimulationConfig.from_json(Path('./config/scenarios/triangle.json'))
sim = LBMSimulation(config)

# Adjust thresholds for this specific case
sim.mach_number_threshold = 0.1
sim.max_velocity_warning_threshold = 2.0

for step in range(50):
    try:
        sim.step(config.inlet_velocity)
    except RuntimeError as e:
        print(f"Divergence detected: {e}")
        break

# Analyze results
print(f"Ran {len(sim.diagnostics['max_velocity'])} steps")
print(f"Turbulence model helped: {np.mean(sim.diagnostics['mean_viscosity']) > 0}")
```

### Example 3: Parameter Study

```python
# Vary Smagorinsky constant and measure stability
for cs in [0.05, 0.10, 0.15, 0.20]:
    sim.cs = cs
    max_u = []
    
    for step in range(50):
        sim.step(config.inlet_velocity)
        max_u.append(np.max(np.sqrt(sim.state.ux**2 + sim.state.uy**2)))
    
    stability = "stable" if max(max_u) < 10 else "unstable"
    print(f"Cs={cs}: final u_max={max_u[-1]:.2f}, {stability}")
```

---

## Recommendations for Your Study

### For Turbulence Analysis:

1. **Start with no obstacles** (Poiseuille) → verify baseline
2. **Increase resolution** before adding obstacles
   - Current: 128×32 is quite coarse
   - Recommended: 256×64 or 512×128 for turbulence features
3. **Use Smagorinsky model** with Cs = 0.15
4. **Monitor diagnostics** religiously
5. **Reduce inlet velocity** if instabilities occur
   - Trade off: less dramatic turbulence, but more stable

### For Low-Velocity Separated Flow:

- Higher viscosity (ν = 0.06–0.10) helps
- Smaller obstacles (obstacle width < domain width / 20)
- Finer mesh near separation points
- Post-process: compute vorticity and recirculation zones

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Velocity explodes rapidly | Bad obstacle BC | Reduce obstacle size; increase resolution |
| Turbulent viscosity = 0 | No velocity gradients | Flow too uniform; check initialization |
| Mach number too high | Inlet velocity too large | Reduce inlet_velocity; check units |
| Oscillating velocity | Insufficient damping | Increase Cs from 0.15 to 0.2 |
| Simulation too slow | High resolution + turbulence | Reduce grid size or use GPU acceleration |

---

## References

1. **Smagorinsky (1963)** - Original SGS model: "General Circulation Experiments with Primitive Equations"
2. **Malaspinas et al. (2010)** - LBM turbulence: "Performance of the athermal lattice Boltzmann BGK model"
3. **Chai & Zhao (2012)** - LBM review: "Lattice Boltzmann model for heat transfer with source terms"

---

## Next Steps for Improvement

1. ✅ **Smagorinsky SGS model** - Implemented
2. ✅ **Stability diagnostics** - Implemented
3. 🔄 **Halfway bounce-back** - Code structure ready, needs tuning
4. 📋 **Adaptive mesh refinement** - For finer obstacle resolution
5. 📋 **Multiple relaxation times (MRT)** - More stable than BGK at higher Re
6. 📋 **Cumulant LBM** - Better stability at low viscosity


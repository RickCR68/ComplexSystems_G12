# Complex System Simulation - Group 12

A Lattice Boltzmann Method (LBM) simulation of fluid dynamics focusing on F1 car's Ground effects and the resulting turbulence, for complexity analysis.

> [!IMPORTANT]
> Testing was done mostly manually with images and animations, as some numerical instability/wrong behaviour was easier to identify through visual aid than most numerical results

## Overview

This project implements a D2Q9 Lattice Boltzmann solver with:
- **Turbulence modeling** using Smagorinsky Large Eddy Simulation (LES)
- **Force calculations** (lift and drag)
- **High-Reynolds turbulent flow simulation** (Re = 1,000 - 25,000)
- **Chaos analysis** and spectral analysis tools

## Setup

```bash
# Clone or navigate to project directory
cd /path/to/G12

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run in Jupyter Notebook (Recommended)

```bash
jupyter lab simulation/main.ipynb
```

Execute the cells to:
1. Configure simulation parameters (Reynolds number, grid size, etc.);
2. Run the simulation with real-time visualization;
3. Generate video frames and comprehensive analysis
    - Recommend FFMPEG for generating videos after image generation;

## Configuration

Key parameters in [main.ipynb](simulation/main.ipynb):

```python
NX, NY = 900, 200        # Grid resolution
REYNOLDS = 5_000         # Reynolds number (flow regime)
FRAMES = 1200            # Total animation frames
STEPS_PER_FRAME = 50     # Physics steps per frame
U_INLET = 0.08           # Inlet velocity (wind tunnel speed)
CS_SMAG = 0.17           # Smagorinsky constant
```

## Physics Details

- **Solver**: D2Q9 Lattice Boltzmann with BGK collision
- **Turbulence**: Smagorinsky LES model
- **Boundary Conditions**: 
  - No-slip walls (bounce-back)
  - Inlet: Constant velocity
  - Outlet: Zero-gradient
- **Force Calculation**: Momentum exchange method at solid boundaries

## Output

The simulation generates timestamped output folders with:

- **PNG frames** for each simulation step
- **Velocity magnitude** visualizations
- **Pressure distribution** heatmaps
- **Streamline plots**
- **Force history graphs** (lift and drag over time)

## Analysis Tools

To run a full analysis suite, run the `complete_analysis` notebook, which creates a timestamped folder with:

- **Chaos metrics**: Lyapunov exponents, correlation dimension
- **Spectral analysis**: FFT-based frequency analysis
- **Phase space reconstruction**: Time-delay embedding
- **Complexity dashboard**: Multi-metric visualization

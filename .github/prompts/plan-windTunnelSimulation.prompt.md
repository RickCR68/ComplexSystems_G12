# Plan: Organize 4-Person Team for Week-Long Wind Tunnel Simulation (Refined)

**TL;DR**: Structure a 7-day sprint with modular .py files for simulation engine, testing, and config management, while Jupyter notebooks serve as the user-facing interface for setup, execution, and analysis. Implement a results cache system using JSON-backed dataclasses in `/results/` to avoid recomputing expensive simulations. Test coverage target: ~60-70% on core solver and boundary condition logic. Roles: lead architect, two core developers (solver + geometry/boundaries), junior on visualizations/caching/docs with mentorship.

## Steps

1. **Define project structure (Day 1 AM)**: Create directories for core modules (`simulation/`, `tests/`, `config/`), notebooks (`notebooks/`), and results cache (`results/`). Design JSON dataclass schema for storing grid states, velocity/vorticity fields, and metadata (Reynolds number, iteration count, wall-clock time).

2. **Implement core modules with test stubs (Days 1-2)**: Dev A writes `simulation/solver.py` (D2Q9 collision/streaming logic) + `tests/test_solver.py` (unit tests for macroscopic recovery); Dev B writes `simulation/geometry.py` (domain/obstacle setup) + `tests/test_geometry.py` (boundary placement validation); Lead creates `config/parameters.py` (JSON-backed dataclass) + `tests/test_config.py`.

3. **Build caching layer (Day 2)**: Implement `simulation/cache.py` using dataclass serialization (json.dumps/json.load) to save/load simulation snapshots to `/results/<scenario_name>_<timestamp>.json`. Cache key: (grid_size, viscosity, inlet_velocity, obstacle_config) → avoid redundant runs.

4. **Create Poiseuille validation notebook (Days 2-3)**: Jupyter notebook (`notebooks/01_poiseuille_validation.ipynb`) orchestrates solver, runs 5000+ iterations, compares velocity profile vs. analytical solution, saves results to `/results/`. Include test assertions (velocity tolerance < 1%, max error visualization).

5. **Extend to cylinder flow notebook (Days 4-5)**: `notebooks/02_cylinder_flow.ipynb` parameterizes obstacle geometry, computes drag coefficient, logs to `/results/`, plots velocity/vorticity with Matplotlib. Reuse cached Poiseuille if parameters match.

6. **Visualization + documentation (Days 5-7)**: Junior dev builds `simulation/visualization.py` (heatmaps, quiver plots, streamlines) tested via `tests/test_visualization.py`, writes API docs, creates usage examples in notebooks. Lead integrates caching into post-processing workflows (load cached results, regenerate plots without recompute).

7. **Daily testing check**: Run test suite (pytest) each morning; enforce >60% coverage on `simulation/` modules using pytest-cov. Failed tests block notebook execution until fixed.

## Further Considerations

### 1. Results caching strategy
Store both raw simulation state (distribution functions) AND derived metrics (velocity fields, drag coefficient) in JSON?

**Recommendation**: Cache velocity/vorticity fields + metadata (cheaper to reload), regenerate other derived quantities on-demand from cached velocity. Saves disk space and caching complexity.

### 2. Test structure
Unit tests for solver correctness (macroscopic recovery), integration tests for full Poiseuille pipeline, validation tests comparing vs. literature (part of notebook assertions, not pytest)?

**Recommendation**: pytest for unit/integration (solver, geometry, config), validation checks embedded in notebooks with visual comparison plots as proof.

### 3. Notebook as executable documentation
Should notebooks include cell magic (`%run`) to import modules OR explicit import statements?

**Recommendation**: Explicit imports + clear cell structure (Setup → Run Sim → Load Cache → Visualize → Export), so notebooks double as usage documentation.

### 4. Config file format
YAML or JSON for parameter files?

**Recommendation**: JSON to keep dependency light; structure mirrors Python dataclass fields, e.g. `{"grid_size": 128, "viscosity": 0.01, "inlet_velocity": 0.1, "obstacles": [{"x": 64, "y": 64, "radius": 8}]}`.

### 5. Extensibility markers
Document in code comments which functions should be modified for 3D/turbulence/thermal (e.g., `# TODO: Extend for D3Q19 lattice` in solver.py)?

**Recommendation**: Yes — add `EXTENSIBILITY.md` file listing 5-10 future hooks (turbulence collision operators, pressure-driven flow, obstacle library), so next team picks up smoothly.

## Project Structure

```
wind-tunnel-sim/
├── simulation/
│   ├── __init__.py
│   ├── solver.py           # D2Q9 LBM solver (core)
│   ├── geometry.py         # Domain & obstacle setup
│   ├── boundaries.py       # BC handling (no-slip, inlet/outlet)
│   ├── cache.py            # JSON-backed result caching
│   └── visualization.py    # Matplotlib plotting utilities
├── config/
│   ├── __init__.py
│   ├── parameters.py       # JSON-backed dataclass config
│   └── scenarios/
│       ├── poiseuille.json # Default Poiseuille params
│       └── cylinder.json   # Default cylinder params
├── tests/
│   ├── test_solver.py      # Unit tests (~60 lines)
│   ├── test_geometry.py    # Unit tests (~40 lines)
│   ├── test_config.py      # Unit tests (~30 lines)
│   └── test_visualization.py # Unit tests (~30 lines)
├── notebooks/
│   ├── 01_poiseuille_validation.ipynb  # Executable workflow
│   └── 02_cylinder_flow.ipynb          # Extended scenario
├── results/
│   ├── poiseuille_2025-01-20_run1.json # Cached simulation
│   └── cylinder_2025-01-20_run1.json
├── EXTENSIBILITY.md        # Future feature roadmap
├── requirements.txt
└── README.md
```

## Roles & Responsibilities

### Lead Architect (Most Experienced)
- Owns overall code structure, API design, and lbmpy integration
- Unblocks library issues and dependency problems
- Integrates modules from other devs into cohesive pipeline
- Final review of solver.py and boundary condition logic
- Mentors junior dev on testing and documentation

### Dev A (Mid-Level, Core Solver)
- Implements `simulation/solver.py` (D2Q9 collision/streaming, macroscopic recovery)
- Writes comprehensive unit tests for solver correctness
- Validates against LBM theory (Chapman-Enskog analysis)

### Dev B (Mid-Level, Geometry & Boundaries)
- Implements `simulation/geometry.py` (rectangular domain, obstacle placement)
- Implements `simulation/boundaries.py` (no-slip walls, inlet/outlet conditions)
- Writes unit tests for geometric boundary placement validation
- Works closely with Dev A on solver-boundary interface

### Junior Dev (Visualization & Documentation)
- Implements `simulation/cache.py` (JSON serialization, result storage)
- Implements `simulation/visualization.py` (Matplotlib utilities)
- Writes Jupyter notebooks as executable examples
- Creates API documentation and usage guides
- Receives pair-programming mentorship from Lead on test coverage

## Timeline Summary

| Phase | Days | Owners | Deliverables |
|-------|------|--------|--------------|
| **Foundation** | 1-2 | All | Project structure, core module stubs, test framework |
| **Core Implementation** | 2-4 | Dev A, Dev B, Lead | solver.py, geometry.py, boundaries.py, caching layer, Poiseuille validation notebook |
| **Extension & Validation** | 4-5 | Dev A, Dev B | Cylinder flow geometry, drag coefficient computation, cached results |
| **Visualization & Polish** | 5-7 | Junior + Lead | visualization.py, documentation, example notebooks, EXTENSIBILITY.md |
| **Testing & QA** | Daily | All | Pytest coverage >60%, notebook execution validation |

## Success Criteria

- ✅ Poiseuille validation passes (velocity tolerance < 1% vs. analytical)
- ✅ Cylinder flow produces measurable drag coefficient within expected range
- ✅ All core modules have >60% test coverage (pytest-cov)
- ✅ Results cached in `/results/` and reloadable without recompute
- ✅ Two working Jupyter notebooks demonstrating full workflow
- ✅ Code documented with EXTENSIBILITY markers for future development
- ✅ All tests pass before end of Day 7

## Notes

- **2D-only scope**: Stick to D2Q9 lattice; 3D (D3Q19) only if ahead of schedule (unlikely)
- **Cylinder as optional**: Poiseuille is hard requirement; cylinder flow is secondary
- **Dependency on lbmpy**: Have manual NumPy fallback if lbmpy installation fails
- **Notebook execution**: Embed test assertions to catch broken code early
- **JSON caching**: Keep format simple (numpy-serializable floats only, avoid complex objects)

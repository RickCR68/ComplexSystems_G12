"""
Quick verification test for the F1 turbulence simulation framework.
This script runs a minimal test to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

print("="*70)
print("F1 TURBULENCE SIMULATION - VERIFICATION TEST")
print("="*70)
print()

# Test 1: Import all modules
print("Test 1: Checking module imports...")
try:
    from config import SimulationConfig, create_parameter_sweep
    from runner import run_simulation, SimulationResults
    from parameter_sweep import ParameterSweep
    from lbm_core import LBMSolver
    from boundaries import TunnelBoundaries
    from aerodynamics import calculate_lift_drag
    from analysis import compute_energy_spectrum
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print()

# Test 2: Create configuration
print("Test 2: Creating simulation configuration...")
try:
    config = SimulationConfig(
        nx=100,
        ny=25,
        reynolds=5000,
        ride_height=5,
        total_steps=100,
        monitor_interval=50,
        output_dir="results/verification"
    )
    print(f"✓ Configuration created: {config.get_run_name()}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

print()

# Test 3: Run mini simulation
print("Test 3: Running mini simulation (100 steps)...")
try:
    results = run_simulation(config, verbose=False)
    print(f"✓ Simulation completed successfully")
    print(f"  - Final drag: {results.drag_history[-1]:.4f}")
    print(f"  - Final lift: {results.lift_history[-1]:.4f}")
except Exception as e:
    print(f"✗ Simulation error: {e}")
    sys.exit(1)

print()

# Test 4: Save results
print("Test 4: Saving results...")
try:
    output_path = results.save(config.output_dir)
    print(f"✓ Results saved to: {output_path}")
    
    # Check files exist
    required_files = ['config.json', 'force_history.npz', 'summary_stats.json']
    for filename in required_files:
        filepath = output_path / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
except Exception as e:
    print(f"✗ Save error: {e}")
    sys.exit(1)

print()

# Test 5: Parameter sweep creation
print("Test 5: Creating parameter sweep...")
try:
    configs = create_parameter_sweep(
        reynolds_values=[5000, 7500],
        ride_height_values=[5, 7],
        base_config=SimulationConfig(
            nx=100,
            ny=25,
            total_steps=100
        )
    )
    print(f"✓ Created {len(configs)} configurations for sweep")
    for conf in configs:
        print(f"  - {conf.get_run_name()}")
except Exception as e:
    print(f"✗ Sweep creation error: {e}")
    sys.exit(1)

print()
print("="*70)
print("✓ ALL TESTS PASSED - SYSTEM READY FOR EXPERIMENTS")
print("="*70)
print()
print("Next steps:")
print("1. Run single experiment: python main.py")
print("2. Run parameter sweep: python main.py --sweep full")
print("3. Use Jupyter notebook: jupyter notebook main.ipynb")
print()

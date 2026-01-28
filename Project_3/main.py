import matplotlib.pyplot as plt
import numpy as np
from lbm_core import LBMSolver
from boundaries import TunnelBoundaries
from analysis import plot_complexity_dashboard
from aerodynamics import calculate_lift_drag, check_ground_effect

# --- CONFIGURATION ---
NX, NY = 400, 100
REYNOLDS = 10000        # High Re to force turbulence
GROUND_TYPE = "no_slip" 

# --- SETUP ---
print(f"Initializing Virtual Wind Tunnel (Re={REYNOLDS}, Ground={GROUND_TYPE})...")
solver = LBMSolver(NX, NY, REYNOLDS, u_inlet=0.1)
bounds = TunnelBoundaries(NX, NY)

bounds.add_ground(type=GROUND_TYPE)
bounds.add_f1_wing_proxy()

# --- HISTORY LISTS (For the Plot) ---
drag_history = []
lift_history = []
step_history = []

# --- MAIN LOOP ---
print("Starting Simulation...")
TOTAL_STEPS = 10000

for step in range(TOTAL_STEPS + 1):
    
    # 1. PHYSICS
    solver.collide_and_stream(bounds.mask)
    bounds.apply_inlet_outlet(solver)
    
    # 2. AERO MONITOR & LOGGING
    if step % 100 == 0:
        fx, fy = calculate_lift_drag(solver, bounds)
        check_ground_effect(fx, fy)
        print(f"Step {step}/{TOTAL_STEPS}")
        
        # Save for plotting
        drag_history.append(fx)
        lift_history.append(fy)
        step_history.append(step)
        
    # 3. COMPLEXITY DASHBOARD
    if step > 0 and step % 500 == 0:
        print(f"   --> Displaying Complexity Dashboard (Step {step})...")
        plot_complexity_dashboard(solver, step)

# --- FINAL VISUALIZATION 1: FLOW FIELD ---
print("Simulation Complete. Saving Visualizations...")

plt.figure(figsize=(12, 5), dpi=150)
velocity_mag = np.sqrt(solver.u[:,:,0]**2 + solver.u[:,:,1]**2)
velocity_mag[bounds.mask] = np.nan 
plt.imshow(velocity_mag, origin='lower', cmap='magma')
plt.colorbar(label="Flow Velocity |u|")
plt.title(f"Final Flow State (Re={REYNOLDS})")
plt.savefig("final_flow.png")
plt.show()

# --- FINAL VISUALIZATION 2: AERO FORCES (NEW!) ---
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(step_history, drag_history, 'r-', label='Drag (Fx)')
plt.plot(step_history, lift_history, 'b-', label='Lift (Fy)')
plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Zero line
plt.xlabel("Simulation Step")
plt.ylabel("Force (LBM Units)")
plt.title(f"Aerodynamic Load History\n(Negative Lift = Downforce)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("aero_forces_history.png") # Send this to your group!
print("Saved 'aero_forces_history.png'.")
plt.show()
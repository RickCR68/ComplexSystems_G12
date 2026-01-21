import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lbm_core import LBMSolver
from boundaries import TunnelBoundaries
from aerodynamics import calculate_lift_drag

# --- CONFIGURATION ---
NX, NY = 400, 100
REYNOLDS = 10000        # High speed turbulence
FRAMES = 300            # Total video frames
STEPS_PER_FRAME = 20    # Speed up factor (sim steps per video frame)

# --- SETUP SIMULATION ---
print("Initializing Simulation for Video Rendering...")
solver = LBMSolver(NX, NY, REYNOLDS, u_inlet=0.1)
bounds = TunnelBoundaries(NX, NY)
bounds.add_ground(type="no_slip")
bounds.add_f1_wing_proxy()

# --- SETUP PLOTS ---
fig = plt.figure(figsize=(10, 8), dpi=100)
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

# Plot 1: The Flow (Top)
ax_flow = fig.add_subplot(gs[0])
# Initialize with zero velocity
velocity_mag = np.zeros((NY, NX))
velocity_mag[bounds.mask] = np.nan
im_flow = ax_flow.imshow(velocity_mag, origin='lower', cmap='magma', vmin=0, vmax=0.15)
ax_flow.set_title(f"F1 Ground Effect Turbulence (Re={REYNOLDS})")
ax_flow.set_xticks([]) # Hide x-axis for cleaner look
plt.colorbar(im_flow, ax=ax_flow, label="Speed |u|", orientation="horizontal", pad=0.2)

# Plot 2: The Force Monitor (Bottom)
ax_force = fig.add_subplot(gs[1])
line_lift, = ax_force.plot([], [], 'b-', linewidth=2, label='Lift (Downforce)')
line_drag, = ax_force.plot([], [], 'r-', linewidth=2, label='Drag')
ax_force.set_xlim(0, FRAMES)
ax_force.set_ylim(-3.0, 1.0) # Adjust based on your previous logs
ax_force.set_xlabel("Video Frame")
ax_force.set_ylabel("Force")
ax_force.legend(loc="upper right")
ax_force.grid(True, alpha=0.3)
ax_force.set_title("Real-Time Aerodynamic Load")

# Data containers
lift_data = []
drag_data = []
frame_indices = []

def update(frame):
    # Run physics steps
    for _ in range(STEPS_PER_FRAME):
        solver.collide_and_stream(bounds.mask)
        bounds.apply_inlet_outlet(solver)
    
    # Calculate Aero
    fx, fy = calculate_lift_drag(solver, bounds)
    
    # Update Data
    lift_data.append(fy)
    drag_data.append(fx)
    frame_indices.append(frame)
    
    # Update Flow Image
    v_mag = np.sqrt(solver.u[:,:,0]**2 + solver.u[:,:,1]**2)
    v_mag[bounds.mask] = np.nan
    im_flow.set_array(v_mag)
    
    # Update Force Lines
    line_lift.set_data(frame_indices, lift_data)
    line_drag.set_data(frame_indices, drag_data)
    
    if frame % 10 == 0:
        print(f"Rendering Frame {frame}/{FRAMES}")
    
    return im_flow, line_lift, line_drag

print("Starting Render... (This might take a minute)")
anim = FuncAnimation(fig, update, frames=FRAMES, interval=50, blit=False)

# Save as MP4
anim.save('f1_turbulence_simulation.gif', writer='pillow', fps=30)
print("Done! Saved 'f1_turbulence_simulation.gif'")
import numpy as np

def calculate_lift_drag(solver, bounds, x_start=50, x_end=80, y_start=5, y_end=30):
    """
    Calculates Drag (Fx) and Lift (Fy) forces acting on the obstacle 
    using the Momentum Exchange Method.
    
    Returns:
        Fx (Drag), Fy (Lift) in simulation units.
    """
    # 1. Identify Boundary Nodes (Fluid cells next to Obstacle cells)
    # This is a simplified neighbor search. 
    # In optimized code, we pre-calculate this list.
    
    fx = 0.0
    fy = 0.0
    
    # Directions (c) and Opposites (noslip)
    c = solver.c
    noslip = solver.noslip
    
    # Iterate over the domain (Slow in Python, but fine for monitoring)
    # OPTIMIZATION: We restrict loop to the "Wing Box" defined in boundaries.py
    # x: 50 to 80, y: 5 to 30
    wing_slice_y = slice(y_start, y_end)
    wing_slice_x = slice(x_start, x_end)
    
    # Local arrays to speed up access
    mask_local = bounds.mask[wing_slice_y, wing_slice_x]
    f_local = solver.f[wing_slice_y, wing_slice_x, :]
    
    # Iterate only where mask is True (The Obstacle)
    solid_y, solid_x = np.where(mask_local)
    
    # Adjust indices to global frame
    for local_y, local_x in zip(solid_y, solid_x):
        # For each solid cell, check all 9 neighbors
        for i in range(1, 9): # Skip 0 (rest)
            # Neighbor coordinates
            next_y = local_y + c[i, 1]
            next_x = local_x + c[i, 0]
            
            # Check if neighbor is within slice bounds
            if (0 <= next_y < mask_local.shape[0]) and (0 <= next_x < mask_local.shape[1]):
                # If neighbor is FLUID (False in mask), momentum is exchanged
                if not mask_local[next_y, next_x]:
                    # The incoming particle direction is opposite to i
                    inv_i = noslip[i]
                    
                    # Force contribution = 2 * f_in (Simple Bounce-Back Rule)
                    # We grab the population moving INTO the wall from the fluid
                    f_val = solver.f[wing_slice_y.start + next_y, 
                                     wing_slice_x.start + next_x, 
                                     inv_i]
                    
                    # F = sum(2 * f * c)
                    momentum = 2.0 * f_val
                    fx += momentum * c[i, 0]
                    fy += momentum * c[i, 1]
                    
    return fx, fy

def check_ground_effect(fx, fy):
    """Interprets the force. Negative Lift = Downforce."""
    print(f"   [AERO] Drag: {fx:.4f} | Lift: {fy:.4f}")
    if fy < 0:
        print("   --> DOWNFORCE GENERATED (Ground Effect Active)")
    else:
        print("   --> LIFT GENERATED (Car is flying!)")
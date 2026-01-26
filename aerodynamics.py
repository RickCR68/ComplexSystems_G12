import numpy as np

class BoundaryForceCalculator:
    """
    Pre-computes boundary cell information for fast force calculation.
    """

    def __init__(self, bounds, solver, x_start=50, x_end=80, y_start=5, y_end=30):
        """
        Pre-compute all boundary-fluid interface pairs once during initialization.
        
        Args:
            bounds: TunnelBoundaries object with obstacle mask
            solver: LBMSolver object (for direction vectors)
            x_start, x_end, y_start, y_end: Region of interest for force calculation
        """
        self.solver = solver
        self.c = solver.c
        self.noslip = solver.noslip
        
        # Define region of interest
        self.y_slice = slice(y_start, y_end)
        self.x_slice = slice(x_start, x_end)
        
        # Extract local mask
        mask_local = bounds.mask[self.y_slice, self.x_slice]
        
        # Find ALL solid cells in the region
        solid_y_local, solid_x_local = np.where(mask_local)
        
        # Pre-compute all boundary-fluid interfaces
        self.boundary_pairs = []  # Will store (global_y, global_x, inv_i, c_i)
        
        for local_y, local_x in zip(solid_y_local, solid_x_local):
            # Check all 9 directions
            for i in range(1, 9):  # Skip rest population (i=0)
                # Neighbor in local coordinates
                next_y_local = local_y + self.c[i, 1]
                next_x_local = local_x + self.c[i, 0]
                
                # Check bounds in local frame
                if (0 <= next_y_local < mask_local.shape[0]) and (0 <= next_x_local < mask_local.shape[1]):
                    # Check if neighbor is FLUID
                    if not mask_local[next_y_local, next_x_local]:
                        # Convert to global coordinates
                        global_y = self.y_slice.start + next_y_local
                        global_x = self.x_slice.start + next_x_local
                        
                        # Incoming direction (opposite to i)
                        inv_i = self.noslip[i]
                        
                        # Store: (where to get f, which direction, momentum factor)
                        self.boundary_pairs.append((global_y, global_x, inv_i, i))
        
        # Convert to arrays for vectorized access
        if self.boundary_pairs:
            self.boundary_pairs = np.array(self.boundary_pairs, dtype=np.int32)
            self.n_pairs = len(self.boundary_pairs)
        else:
            self.n_pairs = 0
    
    def calculate(self):
        """
        Vectorized force calculation using pre-computed boundary pairs.
        Much faster than the original nested loop approach.
        
        Returns:
            fx (Drag), fy (Lift) in simulation units
        """
        if self.n_pairs == 0:
            return 0.0, 0.0
        
        # Extract all boundary cell indices
        y_indices = self.boundary_pairs[:, 0]
        x_indices = self.boundary_pairs[:, 1]
        inv_i_indices = self.boundary_pairs[:, 2]
        i_indices = self.boundary_pairs[:, 3]
        
        # Vectorized: Get all f values at once
        # solver.f shape: (ny, nx, 9)
        f_vals = self.solver.f[y_indices, x_indices, inv_i_indices]  # Shape: (n_pairs,)
        
        # Vectorized: Compute momentum contributions
        momentum = 2.0 * f_vals  # Shape: (n_pairs,)
        
        # Vectorized: Sum forces in x and y directions
        # c[i_indices, 0] gives x-component for each pair
        # c[i_indices, 1] gives y-component for each pair
        fx = np.sum(momentum * self.c[i_indices, 0])
        fy = np.sum(momentum * self.c[i_indices, 1])
        
        return fx, fy


def calculate_lift_drag(solver, bounds, x_start=50, x_end=80, y_start=5, y_end=30):
    """
    Backward-compatible wrapper for old function signature.
    Creates calculator on-the-fly if needed (slower).
    For repeated calls, create BoundaryForceCalculator once in initialization.
    
    Returns:
        Fx (Drag), Fy (Lift) in simulation units.
    """
    calc = BoundaryForceCalculator(bounds, solver, x_start, x_end, y_start, y_end)
    return calc.calculate()

def check_ground_effect(fx, fy):
    """Interprets the force. Negative Lift = Downforce."""
    print(f"   [AERO] Drag: {fx:.4f} | Lift: {fy:.4f}")
    if fy < 0:
        print("   --> DOWNFORCE GENERATED (Ground Effect Active)")
    else:
        print("   --> LIFT GENERATED (Car is flying!)")
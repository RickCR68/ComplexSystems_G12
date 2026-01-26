import numpy as np

class TunnelBoundaries:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.mask = np.zeros((ny, nx), dtype=bool)

    def add_ground(self, type="no_slip"):
        """
        Adds the floor.
        type="no_slip": Real track (friction).
        type="slip": Ideal wind tunnel (no friction).
        """
        if type == "no_slip":
            self.mask[0, :] = True # Bottom row is solid
    
    def add_f1_wing_proxy(self, x_pos=50, height=5, length=30, slope=0.5):
        """
        Creates a triangular proxy for the F1 wing.
        Located near the ground to trigger Ground Effect.
        """
        y, x = np.ogrid[:self.ny, :self.nx]
        # Triangle shape logic: (x > 50) and (x < 80) and (y < slope)
        wing_shape = (x > x_pos) & (x < x_pos + length) & (y > height) & (y < height + (x - x_pos) * slope)
        self.mask |= wing_shape # Combine with existing mask

    def add_reverse_triangle(self, x_pos=50, height=5, length=30, slope=0.5):
        """
        Creates a triangular proxy for the F1 wing.
        Located near the ground to trigger Ground Effect.
        """
        y, x = np.ogrid[:self.ny, :self.nx]
        # Triangle shape logic: (x > 50) and (x < 80) and (y > slope)
        wing_shape = (x > x_pos) & (x < x_pos + length) & (y > height) & (y < height + (length - (x - x_pos)) * slope)
        self.mask |= wing_shape # Combine with existing mask

    def add_rectangular_obstacle(self, x_start, y_start, length, height):
        """
        Adds a square obstacle to the tunnel.
        """
        y, x = np.ogrid[:self.ny, :self.nx]
        obstacle_shape = (x >= x_start) & (x < x_start + length) & (y >= y_start) & (y < y_start + height)
        self.mask |= obstacle_shape # Combine with existing mask

    def apply_inlet_outlet(self, solver):
        """
        Fixed-velocity inlet and improved outlet BC.
        - Inlet: Sets equilibrium distribution with specified velocity
        - Outlet: Convective outlet (prevents reflections)
        """
        # INLET: Create proper shapes for equilibrium calculation
        # All inlet cells have rho=1.0 and velocity=[u_inlet, 0]
        rho_inlet = np.ones((self.ny, 1))  # Shape: (ny, 1)
        u_inlet = np.zeros((self.ny, 1, 2))  # Shape: (ny, 1, 2)
        u_inlet[:, :, 0] = solver.u_inlet  # x-component
        # y-component stays zero
        
        # Compute equilibrium and assign to inlet column
        f_inlet = solver.equilibrium(rho_inlet, u_inlet)  # Shape: (ny, 1, 9)
        solver.f[:, 0, :] = f_inlet[:, 0, :]  # Assign properly: (ny, 9)
        
        # OUTLET: Convective Outlet BC (Better than zero-gradient)
        # Principle: Advect the distribution outward without reflection
        # This prevents disturbances from bouncing back into the domain
        # Implementation: Linear extrapolation using upstream two columns
        # f_out = 2*f[-2] - f[-3]
        solver.f[:, -1, :] = 2.0 * solver.f[:, -2, :] - solver.f[:, -3, :]
        
        # Safety clip to prevent instability from extrapolation overshooting
        solver.f[:, -1, :] = np.clip(solver.f[:, -1, :], 0.0, None)
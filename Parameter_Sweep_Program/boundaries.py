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
        Simple Periodic or Fixed Velocity boundaries.
        (For Phase 1, we just force the inlet).
        """
        solver.f[:, 0, :] = solver.equilibrium(1.0, np.array([solver.u_inlet, 0, 0]).reshape(1,1,3))[0,0,:]
        # Simple outlet outflow (zero gradient)
        solver.f[:, -1, :] = solver.f[:, -2, :]
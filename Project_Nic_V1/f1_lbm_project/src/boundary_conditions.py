"""
Boundary conditions for LBM simulations.
"""

import numpy as np
from typing import Literal


class BoundaryConditions:
    """Handle various boundary conditions for LBM."""
    
    def __init__(self, sim):
        """
        Initialize boundary conditions.
        
        Parameters:
        -----------
        sim : LBMSimulation
            Reference to the simulation object
        """
        self.sim = sim
        self.lattice = sim.lattice
        
    def apply_inlet_velocity(self, u_inlet: float, axis: Literal['left', 'right', 'top', 'bottom'] = 'left'):
        """
        Apply velocity inlet boundary condition using Zou-He method.
        
        Parameters:
        -----------
        u_inlet : float
            Inlet velocity
        axis : str
            Which boundary is the inlet
        """
        if axis == 'left':
            x = 0
            # Set velocity
            self.sim.u[0, x, :] = u_inlet
            self.sim.u[1, x, :] = 0
            
            # Compute density from non-equilibrium bounce-back
            self.sim.rho[x, :] = (
                1.0 / (1.0 - u_inlet) * (
                    np.sum(self.sim.f[[0, 2, 4], x, :], axis=0) +
                    2.0 * np.sum(self.sim.f[[3, 6, 7], x, :], axis=0)
                )
            )
            
            # Set unknown distributions
            self.sim.f[1, x, :] = self.sim.f[3, x, :] + 2.0/3.0 * self.sim.rho[x, :] * u_inlet
            self.sim.f[5, x, :] = self.sim.f[7, x, :] + 0.5 * (self.sim.f[4, x, :] - self.sim.f[2, x, :]) + \
                                   1.0/6.0 * self.sim.rho[x, :] * u_inlet
            self.sim.f[8, x, :] = self.sim.f[6, x, :] - 0.5 * (self.sim.f[4, x, :] - self.sim.f[2, x, :]) + \
                                   1.0/6.0 * self.sim.rho[x, :] * u_inlet
                                   
    def apply_outlet_pressure(self, rho_outlet: float = 1.0, axis: Literal['left', 'right', 'top', 'bottom'] = 'right'):
        """
        Apply pressure/density outlet boundary condition.
        
        Parameters:
        -----------
        rho_outlet : float
            Outlet density
        axis : str
            Which boundary is the outlet
        """
        if axis == 'right':
            x = -1
            # Set density
            self.sim.rho[x, :] = rho_outlet
            
            # Extrapolate velocity from interior
            self.sim.u[:, x, :] = self.sim.u[:, x-1, :]
            
            # Compute unknown distributions
            u_out = self.sim.u[0, x, :]
            
            self.sim.f[3, x, :] = self.sim.f[1, x, :] - 2.0/3.0 * rho_outlet * u_out
            self.sim.f[7, x, :] = self.sim.f[5, x, :] + 0.5 * (self.sim.f[4, x, :] - self.sim.f[2, x, :]) - \
                                   1.0/6.0 * rho_outlet * u_out
            self.sim.f[6, x, :] = self.sim.f[8, x, :] - 0.5 * (self.sim.f[4, x, :] - self.sim.f[2, x, :]) - \
                                   1.0/6.0 * rho_outlet * u_out
                                   
    def apply_slip_bottom(self):
        """Apply slip boundary condition at bottom wall (specular reflection)."""
        y = 0
        # Specular reflection for bottom boundary
        # Reflects distributions, maintaining tangential velocity
        self.sim.f[2, :, y] = self.sim.f[4, :, y].copy()
        self.sim.f[5, :, y] = self.sim.f[7, :, y].copy()
        self.sim.f[6, :, y] = self.sim.f[8, :, y].copy()
        
    def apply_no_slip_bottom(self):
        """Apply no-slip boundary condition at bottom wall (bounce-back)."""
        y = 0
        # Bounce-back for no-slip
        self.sim.f[2, :, y] = self.sim.f[4, :, y].copy()
        self.sim.f[5, :, y] = self.sim.f[7, :, y].copy()
        self.sim.f[6, :, y] = self.sim.f[8, :, y].copy()
        
    def apply_slip_top(self):
        """Apply slip boundary condition at top wall."""
        y = -1
        # Specular reflection for top boundary
        self.sim.f[4, :, y] = self.sim.f[2, :, y].copy()
        self.sim.f[7, :, y] = self.sim.f[5, :, y].copy()
        self.sim.f[8, :, y] = self.sim.f[6, :, y].copy()
        
    def apply_periodic_x(self):
        """Apply periodic boundary conditions in x-direction."""
        # Left boundary = Right boundary
        self.sim.f[:, 0, :] = self.sim.f[:, -2, :]
        self.sim.f[:, -1, :] = self.sim.f[:, 1, :]
        
    def apply_periodic_y(self):
        """Apply periodic boundary conditions in y-direction."""
        # Bottom boundary = Top boundary
        self.sim.f[:, :, 0] = self.sim.f[:, :, -2]
        self.sim.f[:, :, -1] = self.sim.f[:, :, 1]


def apply_all_boundaries(
    sim,
    u_inlet: float,
    bottom_type: Literal['slip', 'no_slip'] = 'slip',
    inlet_axis: str = 'left',
    outlet_axis: str = 'right'
):
    """
    Apply all boundary conditions for a typical wind tunnel setup.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    u_inlet : float
        Inlet velocity
    bottom_type : str
        Type of bottom boundary ('slip' or 'no_slip')
    inlet_axis : str
        Inlet location
    outlet_axis : str
        Outlet location
    """
    bc = BoundaryConditions(sim)
    
    # Apply inlet
    bc.apply_inlet_velocity(u_inlet, axis=inlet_axis)
    
    # Apply outlet
    bc.apply_outlet_pressure(rho_outlet=1.0, axis=outlet_axis)
    
    # Apply bottom boundary
    if bottom_type == 'slip':
        bc.apply_slip_bottom()
    else:
        bc.apply_no_slip_bottom()
    
    # Apply top boundary (slip)
    bc.apply_slip_top()
    
    return bc

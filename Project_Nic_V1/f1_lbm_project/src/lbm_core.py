"""
Core Lattice Boltzmann Method implementation using D2Q9 lattice.
"""

import numpy as np
from typing import Tuple, Optional


class D2Q9Lattice:
    """D2Q9 lattice configuration for 2D LBM."""
    
    def __init__(self):
        # Lattice velocities (9 directions)
        self.c = np.array([
            [0,  1,  0, -1,  0,  1, -1, -1,  1],  # x-components
            [0,  0,  1,  0, -1,  1,  1, -1, -1]   # y-components
        ])
        
        # Lattice weights
        self.w = np.array([
            4/9,                          # Center (0)
            1/9, 1/9, 1/9, 1/9,          # Cardinal directions (1-4)
            1/36, 1/36, 1/36, 1/36       # Diagonal directions (5-8)
        ])
        
        # Opposite directions for bounce-back
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
    @property
    def num_velocities(self) -> int:
        return 9


class LBMSimulation:
    """
    Main LBM simulation class for 2D fluid flow.
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        reynolds_number: float,
        u_max: float = 0.1,
        lattice: Optional[D2Q9Lattice] = None
    ):
        """
        Initialize LBM simulation.
        
        Parameters:
        -----------
        nx : int
            Number of lattice points in x-direction
        ny : int
            Number of lattice points in y-direction
        reynolds_number : float
            Reynolds number for the simulation
        u_max : float
            Maximum velocity (in lattice units)
        lattice : D2Q9Lattice, optional
            Lattice configuration
        """
        self.nx = nx
        self.ny = ny
        self.Re = reynolds_number
        self.u_max = u_max
        
        # Initialize lattice
        self.lattice = lattice if lattice is not None else D2Q9Lattice()
        
        # Calculate kinematic viscosity and relaxation time
        self.L = ny  # Characteristic length (channel height)
        self.nu = self.u_max * self.L / self.Re
        self.tau = 3.0 * self.nu + 0.5
        self.omega = 1.0 / self.tau  # Relaxation parameter
        
        # Initialize distribution functions
        self.f = np.zeros((self.lattice.num_velocities, nx, ny))
        self.f_eq = np.zeros_like(self.f)
        self.f_new = np.zeros_like(self.f)
        
        # Macroscopic quantities
        self.rho = np.ones((nx, ny))
        self.u = np.zeros((2, nx, ny))
        
        # Obstacle mask (False = fluid, True = solid)
        self.obstacle = np.zeros((nx, ny), dtype=bool)
        
        # Iteration counter
        self.iteration = 0
        
    def initialize_equilibrium(self, rho_init: float = 1.0, u_init: Optional[np.ndarray] = None):
        """
        Initialize distribution functions to equilibrium.
        
        Parameters:
        -----------
        rho_init : float
            Initial density
        u_init : np.ndarray, optional
            Initial velocity field [2, nx, ny]
        """
        if u_init is None:
            u_init = np.zeros((2, self.nx, self.ny))
            
        self.rho[:] = rho_init
        self.u[:] = u_init
        
        self._compute_equilibrium(self.rho, self.u)
        self.f[:] = self.f_eq[:]
        
    def _compute_equilibrium(self, rho: np.ndarray, u: np.ndarray):
        """
        Compute equilibrium distribution function.
        
        Parameters:
        -----------
        rho : np.ndarray
            Density field
        u : np.ndarray
            Velocity field [2, nx, ny]
        """
        c = self.lattice.c
        w = self.lattice.w
        
        # Compute u·u
        u_sqr = u[0]**2 + u[1]**2
        
        for i in range(self.lattice.num_velocities):
            # c_i · u
            cu = c[0, i] * u[0] + c[1, i] * u[1]
            
            # Equilibrium distribution
            self.f_eq[i] = w[i] * rho * (
                1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sqr
            )
    
    def _compute_macroscopic(self):
        """Compute macroscopic quantities (density and velocity) from distributions."""
        # Density: sum of all distributions
        self.rho = np.sum(self.f, axis=0)
        
        # Momentum: sum of (distribution * lattice velocity)
        c = self.lattice.c
        self.u[0] = np.sum(c[0, :, np.newaxis, np.newaxis] * self.f, axis=0) / self.rho
        self.u[1] = np.sum(c[1, :, np.newaxis, np.newaxis] * self.f, axis=0) / self.rho
        
        # Set velocity to zero in obstacles
        self.u[:, self.obstacle] = 0
        
    def collision(self):
        """BGK collision step."""
        self._compute_macroscopic()
        self._compute_equilibrium(self.rho, self.u)
        
        # BGK collision: f_new = f + omega * (f_eq - f)
        self.f -= self.omega * (self.f - self.f_eq)
        
    def streaming(self):
        """Stream distributions to neighboring nodes."""
        c = self.lattice.c
        
        for i in range(self.lattice.num_velocities):
            self.f[i] = np.roll(np.roll(self.f[i], c[0, i], axis=0), c[1, i], axis=1)
            
    def bounce_back(self):
        """Apply bounce-back boundary condition for obstacles."""
        if not np.any(self.obstacle):
            return
            
        for i in range(self.lattice.num_velocities):
            # Bounce back at obstacle nodes
            f_temp = self.f[i].copy()
            self.f[i][self.obstacle] = self.f[self.lattice.opposite[i]][self.obstacle]
            
    def set_obstacle(self, obstacle_mask: np.ndarray):
        """
        Set obstacle geometry.
        
        Parameters:
        -----------
        obstacle_mask : np.ndarray
            Boolean array where True indicates solid nodes
        """
        self.obstacle = obstacle_mask.copy()
        
    def step(self):
        """Perform one LBM time step."""
        self.collision()
        self.streaming()
        self.bounce_back()
        self.iteration += 1
        
    def run(self, num_steps: int, checkpoint_interval: Optional[int] = None) -> dict:
        """
        Run simulation for a given number of steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to run
        checkpoint_interval : int, optional
            Save data every N steps
            
        Returns:
        --------
        dict : Dictionary containing simulation history
        """
        history = {
            'iterations': [],
            'u_mean': [],
            'v_mean': [],
            'rho_mean': []
        }
        
        for step in range(num_steps):
            self.step()
            
            if checkpoint_interval and (step % checkpoint_interval == 0):
                history['iterations'].append(self.iteration)
                history['u_mean'].append(np.mean(self.u[0]))
                history['v_mean'].append(np.mean(self.u[1]))
                history['rho_mean'].append(np.mean(self.rho))
                
        return history
    
    def get_velocity_magnitude(self) -> np.ndarray:
        """Get velocity magnitude field."""
        return np.sqrt(self.u[0]**2 + self.u[1]**2)
    
    def get_vorticity(self) -> np.ndarray:
        """Compute vorticity field (ω = ∂v/∂x - ∂u/∂y)."""
        dvdx = np.gradient(self.u[1], axis=0)
        dudy = np.gradient(self.u[0], axis=1)
        return dvdx - dudy

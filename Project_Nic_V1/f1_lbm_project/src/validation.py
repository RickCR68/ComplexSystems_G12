"""
Validation functions for LBM simulations.
Compare numerical results with analytical solutions.
"""

import numpy as np
from typing import Tuple


def poiseuille_analytical(ny: int, u_max: float) -> np.ndarray:
    """
    Analytical solution for Poiseuille (channel) flow.
    
    Parameters:
    -----------
    ny : int
        Number of lattice points in y-direction
    u_max : float
        Maximum velocity (at channel center)
        
    Returns:
    --------
    np.ndarray : Velocity profile u(y)
    """
    y = np.arange(ny)
    y_center = (ny - 1) / 2.0
    
    # Parabolic profile: u(y) = u_max * (1 - ((y - y_center) / y_center)^2)
    u_analytical = u_max * (1 - ((y - y_center) / y_center)**2)
    
    return u_analytical


def compute_l2_error(numerical: np.ndarray, analytical: np.ndarray) -> float:
    """
    Compute L2 norm error between numerical and analytical solutions.
    
    Parameters:
    -----------
    numerical : np.ndarray
        Numerical solution
    analytical : np.ndarray
        Analytical solution
        
    Returns:
    --------
    float : L2 error
    """
    error = np.sqrt(np.mean((numerical - analytical)**2))
    return error


def validate_poiseuille(sim, u_max: float, sample_x: int = None) -> dict:
    """
    Validate simulation against Poiseuille flow analytical solution.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    u_max : float
        Expected maximum velocity
    sample_x : int, optional
        X position to sample velocity profile (default: middle)
        
    Returns:
    --------
    dict : Contains error, numerical profile, and analytical profile
    """
    if sample_x is None:
        sample_x = sim.nx // 2
    
    # Get numerical velocity profile at sample location
    u_numerical = sim.u[0, sample_x, :]
    
    # Get analytical solution
    u_analytical = poiseuille_analytical(sim.ny, u_max)
    
    # Compute error
    error = compute_l2_error(u_numerical, u_analytical)
    
    return {
        'l2_error': error,
        'u_numerical': u_numerical,
        'u_analytical': u_analytical,
        'relative_error': error / u_max
    }


def cavity_benchmark_data() -> dict:
    """
    Ghia et al. (1982) benchmark data for lid-driven cavity at Re=100.
    
    Returns:
    --------
    dict : Benchmark data for centerline velocities
    """
    # U velocity along vertical centerline (x=0.5)
    y_coords = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 
                         0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 
                         0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    
    u_centerline = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717,
                             0.23151, 0.00332, -0.13641, -0.20581, -0.21090,
                             -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                             -0.03717, 0.00000])
    
    # V velocity along horizontal centerline (y=0.5)
    x_coords = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063,
                         0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563,
                         0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
    
    v_centerline = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
                             -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
                             0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                             0.09233, 0.00000])
    
    return {
        'u': {'y': y_coords, 'u': u_centerline},
        'v': {'x': x_coords, 'v': v_centerline}
    }


def mass_conservation_check(sim) -> float:
    """
    Check mass conservation (total density should be constant).
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
        
    Returns:
    --------
    float : Total mass in domain
    """
    total_mass = np.sum(sim.rho[~sim.obstacle])
    return total_mass


def momentum_conservation_check(sim) -> Tuple[float, float]:
    """
    Check momentum conservation (total momentum should be constant).
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
        
    Returns:
    --------
    tuple : (total_momentum_x, total_momentum_y)
    """
    fluid_mask = ~sim.obstacle
    momentum_x = np.sum(sim.rho[fluid_mask] * sim.u[0][fluid_mask])
    momentum_y = np.sum(sim.rho[fluid_mask] * sim.u[1][fluid_mask])
    
    return momentum_x, momentum_y

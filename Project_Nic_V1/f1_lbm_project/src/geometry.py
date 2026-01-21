"""
Geometry generation for LBM simulations.
"""

import numpy as np
from typing import Tuple


def create_cylinder(
    nx: int,
    ny: int,
    center: Tuple[float, float],
    radius: float
) -> np.ndarray:
    """
    Create a circular cylinder obstacle.
    
    Parameters:
    -----------
    nx : int
        Domain width
    ny : int
        Domain height
    center : tuple
        (x, y) center position
    radius : float
        Cylinder radius
        
    Returns:
    --------
    np.ndarray : Boolean mask (True = solid, False = fluid)
    """
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    cx, cy = center
    
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    obstacle = distance <= radius
    
    return obstacle


def create_rectangle(
    nx: int,
    ny: int,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int
) -> np.ndarray:
    """
    Create a rectangular obstacle.
    
    Parameters:
    -----------
    nx, ny : int
        Domain dimensions
    x_start, x_end : int
        Rectangle bounds in x
    y_start, y_end : int
        Rectangle bounds in y
        
    Returns:
    --------
    np.ndarray : Boolean mask
    """
    obstacle = np.zeros((nx, ny), dtype=bool)
    obstacle[x_start:x_end, y_start:y_end] = True
    return obstacle


def create_naca_airfoil(
    chord: int,
    thickness: float = 0.12,
    camber: float = 0.02,
    camber_position: float = 0.4,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil coordinates.
    
    Parameters:
    -----------
    chord : int
        Chord length in lattice units
    thickness : float
        Maximum thickness as fraction of chord (e.g., 0.12 for 12%)
    camber : float
        Maximum camber as fraction of chord
    camber_position : float
        Position of maximum camber (0-1)
    num_points : int
        Number of points to generate
        
    Returns:
    --------
    tuple : (x_coords, y_coords) arrays for upper and lower surfaces
    """
    # Cosine spacing for better resolution at leading/trailing edges
    beta = np.linspace(0, np.pi, num_points)
    x = chord * (1 - np.cos(beta)) / 2
    
    # Thickness distribution (symmetric airfoil)
    yt = 5 * thickness * chord * (
        0.2969 * np.sqrt(x/chord) -
        0.1260 * (x/chord) -
        0.3516 * (x/chord)**2 +
        0.2843 * (x/chord)**3 -
        0.1015 * (x/chord)**4
    )
    
    # Mean camber line
    yc = np.zeros_like(x)
    if camber > 0:
        m = camber
        p = camber_position
        
        # Forward of maximum camber
        mask1 = x <= p * chord
        yc[mask1] = m * (x[mask1] / (p**2)) * (2*p - x[mask1]/chord)
        
        # Aft of maximum camber
        mask2 = x > p * chord
        yc[mask2] = m * ((chord - x[mask2]) / ((1-p)**2)) * (1 + x[mask2]/chord - 2*p)
    
    # Upper and lower surfaces
    x_upper = x
    y_upper = yc + yt
    
    x_lower = x
    y_lower = yc - yt
    
    return x_upper, y_upper, x_lower, y_lower


def create_simple_wing(
    nx: int,
    ny: int,
    chord: int,
    thickness: int,
    position: Tuple[int, int],
    angle: float = 0
) -> np.ndarray:
    """
    Create a simplified wing shape (rectangular with rounded leading edge).
    
    Parameters:
    -----------
    nx, ny : int
        Domain dimensions
    chord : int
        Wing chord length
    thickness : int
        Wing thickness
    position : tuple
        (x, y) position of leading edge
    angle : float
        Angle of attack in degrees
        
    Returns:
    --------
    np.ndarray : Boolean mask
    """
    obstacle = np.zeros((nx, ny), dtype=bool)
    x_start, y_start = position
    
    # Create basic rectangular wing
    x_end = x_start + chord
    y_end = y_start + thickness
    
    # Ensure within bounds
    x_end = min(x_end, nx)
    y_end = min(y_end, ny)
    
    if angle == 0:
        # Simple rectangle
        obstacle[x_start:x_end, y_start:y_end] = True
    else:
        # Rotate the wing (simplified - uses point rotation)
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        
        # Center of rotation
        cx = x_start + chord // 2
        cy = y_start + thickness // 2
        
        # Rotate coordinates
        angle_rad = np.deg2rad(angle)
        x_rot = (x_grid - cx) * np.cos(angle_rad) + (y_grid - cy) * np.sin(angle_rad) + cx
        y_rot = -(x_grid - cx) * np.sin(angle_rad) + (y_grid - cy) * np.cos(angle_rad) + cy
        
        # Check if points are inside original rectangle
        inside_x = (x_rot >= x_start) & (x_rot < x_end)
        inside_y = (y_rot >= y_start) & (y_rot < y_end)
        obstacle = inside_x & inside_y
    
    return obstacle


def create_f1_wing_proxy(
    nx: int,
    ny: int,
    chord: int,
    ride_height: int,
    thickness_ratio: float = 0.15,
    angle_of_attack: float = 5.0,
    x_position: int = None
) -> np.ndarray:
    """
    Create a simplified F1 front wing proxy.
    
    Parameters:
    -----------
    nx, ny : int
        Domain dimensions
    chord : int
        Wing chord length
    ride_height : int
        Distance from ground to bottom of wing
    thickness_ratio : float
        Thickness as fraction of chord
    angle_of_attack : float
        Angle in degrees
    x_position : int, optional
        X position of leading edge (default: nx//4)
        
    Returns:
    --------
    np.ndarray : Boolean mask
    """
    if x_position is None:
        x_position = nx // 4
    
    thickness = int(chord * thickness_ratio)
    position = (x_position, ride_height)
    
    wing = create_simple_wing(nx, ny, chord, thickness, position, angle=angle_of_attack)
    
    return wing


def get_boundary_nodes(obstacle: np.ndarray) -> np.ndarray:
    """
    Get the boundary nodes of an obstacle (nodes adjacent to fluid).
    
    Parameters:
    -----------
    obstacle : np.ndarray
        Boolean obstacle mask
        
    Returns:
    --------
    np.ndarray : Indices of boundary nodes
    """
    from scipy import ndimage
    
    # Dilate the obstacle
    dilated = ndimage.binary_dilation(obstacle)
    
    # Boundary is where dilated - original
    boundary = dilated & ~obstacle
    
    return boundary

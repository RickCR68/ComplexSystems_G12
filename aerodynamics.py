"""
Aerodynamic Force Calculations

Computes lift and drag forces using the Momentum Exchange Method (MEM).

The MEM calculates forces by summing momentum exchanged at fluid-solid interfaces
during bounce-back operations. This is more accurate than stress integration
for lattice Boltzmann methods.

Reference:
- Mei et al., "Force evaluation in the LBM involving curved geometry" (2002)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings


@dataclass
class AeroForces:
    """
    Container for aerodynamic force data.
    
    Attributes
    ----------
    drag : float
        Drag force (x-direction, positive = resisting flow)
    lift : float
        Lift force (y-direction, positive = upward)
    cd : float
        Drag coefficient (if reference values provided)
    cl : float
        Lift coefficient (if reference values provided)
    """
    drag: float
    lift: float
    cd: float = np.nan
    cl: float = np.nan
    
    @property
    def downforce(self) -> float:
        """Downforce = negative lift."""
        return -self.lift
    
    @property
    def efficiency(self) -> float:
        """Aerodynamic efficiency: |Lift| / Drag."""
        if abs(self.drag) < 1e-10:
            return np.nan
        return abs(self.lift) / abs(self.drag)
    
    def __repr__(self):
        return (f"AeroForces(drag={self.drag:.6f}, lift={self.lift:.6f}, "
                f"Cd={self.cd:.4f}, Cl={self.cl:.4f})")


def calculate_forces_mem(solver, bounds, 
                         reference_area: Optional[float] = None,
                         reference_velocity: Optional[float] = None) -> AeroForces:
    """
    Calculate aerodynamic forces using Momentum Exchange Method.
    
    The MEM computes forces by summing momentum transferred during
    bounce-back at all fluid-solid interfaces:
        F = sum_boundary(2 * f_in * c)
    
    Parameters
    ----------
    solver : LBMSolver
        The LBM solver with current flow state
    bounds : TunnelBoundaries
        Geometry with obstacle mask
    reference_area : float, optional
        Reference area for coefficient calculation.
        If None, uses obstacle height.
    reference_velocity : float, optional
        Reference velocity for coefficient calculation.
        If None, uses inlet velocity.
        
    Returns
    -------
    AeroForces
        Computed drag and lift forces with coefficients
    """
    fx = 0.0
    fy = 0.0
    
    c = solver.c
    noslip = solver.noslip
    
    # Get obstacle region (restrict loop for efficiency)
    y_slice, x_slice = bounds.get_obstacle_slice()
    
    mask_local = bounds.mask[y_slice, x_slice].copy()
    # Exclude ground from force calculation (it's not the car)
    if bounds.config.ground_type in ["no_slip", "moving"]:
        ground_in_slice = max(0, 0 - y_slice.start)
        if 0 <= ground_in_slice < mask_local.shape[0]:
            mask_local[ground_in_slice, :] = False
    
    # Find solid cells
    solid_y, solid_x = np.where(mask_local)
    
    for local_y, local_x in zip(solid_y, solid_x):
        global_y = y_slice.start + local_y
        global_x = x_slice.start + local_x
        
        # Check all 8 streaming directions (skip rest particle i=0)
        for i in range(1, 9):
            # Neighbor in direction i
            ny_neighbor = local_y + c[i, 1]
            nx_neighbor = local_x + c[i, 0]
            
            # Check bounds
            if not (0 <= ny_neighbor < mask_local.shape[0] and 
                    0 <= nx_neighbor < mask_local.shape[1]):
                continue
                
            # If neighbor is fluid (boundary link)
            if not mask_local[ny_neighbor, nx_neighbor]:
                # Get inverse direction
                i_inv = noslip[i]
                
                # Get incoming distribution from fluid cell
                global_ny = y_slice.start + ny_neighbor
                global_nx = x_slice.start + nx_neighbor
                f_in = solver.f[global_ny, global_nx, i_inv]
                
                # Momentum exchange: F = 2 * f_in * c
                momentum = 2.0 * f_in
                fx += momentum * c[i, 0]
                fy += momentum * c[i, 1]
    
    # Compute coefficients
    if reference_area is None and bounds.obstacle_bounds is not None:
        _, _, y_min, y_max = bounds.obstacle_bounds
        reference_area = float(y_max - y_min)
    else:
        reference_area = reference_area or float(bounds.ny)
        
    if reference_velocity is None:
        reference_velocity = solver.u_inlet
        
    # Cd = Fd / (0.5 * rho * U^2 * A)
    # In lattice units, rho = 1
    dynamic_pressure = 0.5 * reference_velocity**2 * reference_area
    
    if dynamic_pressure > 1e-10:
        cd = fx / dynamic_pressure
        cl = fy / dynamic_pressure
    else:
        cd = np.nan
        cl = np.nan
        
    return AeroForces(drag=fx, lift=fy, cd=cd, cl=cl)


def calculate_pressure_distribution(solver, bounds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate pressure distribution on obstacle surface.
    
    Parameters
    ----------
    solver : LBMSolver
        The solver with current state
    bounds : TunnelBoundaries
        The geometry
        
    Returns
    -------
    surface_coords : ndarray
        Coordinates of surface points, shape (n_points, 2)
    pressure : ndarray
        Pressure at each surface point
    """
    # Pressure from equation of state: p = rho * cs^2 = rho / 3
    pressure_field = solver.rho / 3.0
    
    # Find surface cells (solid cells with at least one fluid neighbor)
    y_slice, x_slice = bounds.get_obstacle_slice()
    mask_local = bounds.mask[y_slice, x_slice]
    
    surface_points = []
    surface_pressure = []
    
    solid_y, solid_x = np.where(mask_local)
    
    for ly, lx in zip(solid_y, solid_x):
        is_surface = False
        
        # Check 4-connectivity for surface
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = ly + dy, lx + dx
            if (0 <= ny < mask_local.shape[0] and 
                0 <= nx < mask_local.shape[1] and
                not mask_local[ny, nx]):
                is_surface = True
                break
                
        if is_surface:
            gy = y_slice.start + ly
            gx = x_slice.start + lx
            surface_points.append([gx, gy])
            
            # Average pressure from neighboring fluid cells
            p_neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = gy + dy, gx + dx
                if (0 <= ny < bounds.ny and 0 <= nx < bounds.nx and
                    not bounds.mask[ny, nx]):
                    p_neighbors.append(pressure_field[ny, nx])
                    
            surface_pressure.append(np.mean(p_neighbors) if p_neighbors else np.nan)
            
    return np.array(surface_points), np.array(surface_pressure)


class ForceMonitor:
    """
    Monitor and record aerodynamic forces during simulation.
    
    Tracks force history and provides statistics.
    
    Parameters
    ----------
    window_size : int
        Window size for moving average calculation
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[AeroForces] = []
        self.steps: List[int] = []
        
    def record(self, step: int, forces: AeroForces):
        """Record forces at a simulation step."""
        self.steps.append(step)
        self.history.append(forces)
        
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get force history as arrays.
        
        Returns
        -------
        steps, drag, lift, cd, cl : ndarray
            Arrays of recorded values
        """
        if not self.history:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]))
            
        steps = np.array(self.steps)
        drag = np.array([f.drag for f in self.history])
        lift = np.array([f.lift for f in self.history])
        cd = np.array([f.cd for f in self.history])
        cl = np.array([f.cl for f in self.history])
        
        return steps, drag, lift, cd, cl
    
    def get_mean_forces(self, last_n: Optional[int] = None) -> AeroForces:
        """
        Get mean forces over recent history.
        
        Parameters
        ----------
        last_n : int, optional
            Number of recent samples to average. Default uses window_size.
            
        Returns
        -------
        AeroForces
            Averaged forces
        """
        if not self.history:
            return AeroForces(0, 0)
            
        n = last_n or self.window_size
        recent = self.history[-n:]
        
        return AeroForces(
            drag=np.mean([f.drag for f in recent]),
            lift=np.mean([f.lift for f in recent]),
            cd=np.mean([f.cd for f in recent]),
            cl=np.mean([f.cl for f in recent])
        )
    
    def get_std_forces(self, last_n: Optional[int] = None) -> Tuple[float, float]:
        """Get standard deviation of recent drag and lift."""
        if len(self.history) < 2:
            return (0.0, 0.0)
            
        n = last_n or self.window_size
        recent = self.history[-n:]
        
        return (
            np.std([f.drag for f in recent]),
            np.std([f.lift for f in recent])
        )
    
    def is_converged(self, tolerance: float = 0.01) -> bool:
        """
        Check if forces have converged.
        
        Uses coefficient of variation (std/mean) as convergence metric.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed coefficient of variation
            
        Returns
        -------
        bool
            True if converged
        """
        if len(self.history) < self.window_size:
            return False
            
        mean_forces = self.get_mean_forces()
        std_drag, std_lift = self.get_std_forces()
        
        if abs(mean_forces.drag) < 1e-10 or abs(mean_forces.lift) < 1e-10:
            return False
            
        cv_drag = std_drag / abs(mean_forces.drag)
        cv_lift = std_lift / abs(mean_forces.lift)
        
        return cv_drag < tolerance and cv_lift < tolerance
    
    def clear(self):
        """Clear all recorded history."""
        self.history.clear()
        self.steps.clear()


def interpret_f1_forces(forces: AeroForces, verbose: bool = True) -> dict:
    """
    Interpret aerodynamic forces in F1 context.
    
    Parameters
    ----------
    forces : AeroForces
        Computed forces
    verbose : bool
        Print interpretation
        
    Returns
    -------
    dict
        Interpretation results
    """
    results = {
        "has_downforce": forces.lift < 0,
        "downforce_magnitude": -forces.lift if forces.lift < 0 else 0,
        "drag_magnitude": forces.drag,
        "efficiency": forces.efficiency,
        "ground_effect_active": forces.lift < -0.1  # Threshold for significant downforce
    }
    
    if verbose:
        print(f"=== Aerodynamic Analysis ===")
        print(f"  Drag Force:     {forces.drag:>10.4f}")
        print(f"  Lift Force:     {forces.lift:>10.4f}")
        print(f"  Drag Coeff Cd:  {forces.cd:>10.4f}")
        print(f"  Lift Coeff Cl:  {forces.cl:>10.4f}")
        print(f"  L/D Ratio:      {forces.efficiency:>10.4f}")
        print()
        
        if results["has_downforce"]:
            print(f"  ✓ DOWNFORCE generated: {results['downforce_magnitude']:.4f}")
            if results["ground_effect_active"]:
                print(f"  ✓ Ground Effect appears ACTIVE")
        else:
            print(f"  ✗ No downforce - car producing LIFT (would fly!)")
            
    return results

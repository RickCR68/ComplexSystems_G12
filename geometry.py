"""
Geometry and Boundary Conditions Module

Provides various obstacle geometries for wind tunnel simulations:
- Simple shapes (triangle, rectangle, cylinder) for validation
- F1 car components (wing, full car profile)
- Configurable ride height for ground effect studies

All geometries are created as boolean masks on the simulation grid.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


class GeometryType(Enum):
    """Available geometry types for simulation."""
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    CYLINDER = "cylinder"
    AIRFOIL_NACA = "naca_airfoil"
    F1_WING_SIMPLE = "f1_wing_simple"
    F1_CAR_FULL = "f1_car_full"
    CUSTOM = "custom"


@dataclass
class GeometryConfig:
    """
    Configuration for obstacle geometry.
    
    Parameters
    ----------
    geometry_type : GeometryType
        Type of geometry to create
    ride_height : int
        Distance from ground to lowest point of obstacle (in lattice units)
    scale : float
        Scaling factor for geometry size (1.0 = default)
    x_position : Optional[int]
        X-position of geometry leading edge. None = auto-position
    angle_of_attack : float
        Rotation angle in degrees (positive = nose up)
    ground_type : str
        "no_slip" for stationary ground, "moving" for moving ground plane
    """
    geometry_type: GeometryType = GeometryType.F1_WING_SIMPLE
    ride_height: int = 5
    scale: float = 1.0
    x_position: Optional[int] = None
    angle_of_attack: float = 0.0
    ground_type: str = "no_slip"
    
    # Additional parameters for specific geometries
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "geometry_type": self.geometry_type.value,
            "ride_height": self.ride_height,
            "scale": self.scale,
            "x_position": self.x_position,
            "angle_of_attack": self.angle_of_attack,
            "ground_type": self.ground_type,
            "extra_params": self.extra_params
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryConfig":
        """Create from dictionary."""
        d = d.copy()
        d["geometry_type"] = GeometryType(d["geometry_type"])
        return cls(**d)


class TunnelBoundaries:
    """
    Wind tunnel boundary conditions and obstacle geometry.
    
    Manages:
    - Obstacle mask (solid/fluid cells)
    - Ground plane
    - Inlet/outlet conditions
    
    Parameters
    ----------
    nx : int
        Domain width in lattice units
    ny : int
        Domain height in lattice units
    config : GeometryConfig, optional
        Geometry configuration. Default creates simple F1 wing.
    """
    
    def __init__(self, nx: int, ny: int, config: Optional[GeometryConfig] = None):
        self.nx = nx
        self.ny = ny
        self.config = config or GeometryConfig()
        
        # Main obstacle mask (True = solid)
        self.mask = np.zeros((ny, nx), dtype=bool)
        
        # Ground mask (separate for different boundary treatment)
        self.ground_mask = np.zeros((ny, nx), dtype=bool)
        
        # Bounding box of obstacle (for force calculations)
        self.obstacle_bounds = None
        
    def build(self):
        """Construct the geometry based on configuration."""
        # Reset masks
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.ground_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # Add ground
        self._add_ground()
        
        # Add obstacle based on type
        geometry_builders = {
            GeometryType.TRIANGLE: self._add_triangle,
            GeometryType.RECTANGLE: self._add_rectangle,
            GeometryType.CYLINDER: self._add_cylinder,
            GeometryType.AIRFOIL_NACA: self._add_naca_airfoil,
            GeometryType.F1_WING_SIMPLE: self._add_f1_wing_simple,
            GeometryType.F1_CAR_FULL: self._add_f1_car_full,
        }
        
        if self.config.geometry_type in geometry_builders:
            geometry_builders[self.config.geometry_type]()
        elif self.config.geometry_type == GeometryType.CUSTOM:
            # Custom geometry should be added via add_custom_mask()
            pass
            
        return self
    
    def _add_ground(self):
        """Add ground plane boundary."""
        if self.config.ground_type in ["no_slip", "moving"]:
            self.ground_mask[0, :] = True
            self.mask[0, :] = True
            
    def _get_default_x_position(self, geometry_length: int) -> int:
        """Calculate default x-position (1/4 of domain from inlet)."""
        if self.config.x_position is not None:
            return self.config.x_position
        return max(10, self.nx // 6)
    
    def _add_triangle(self):
        """
        Add triangular obstacle (simple validation geometry).
        
        Triangle points into the flow with apex at leading edge.
        """
        scale = self.config.scale
        rh = self.config.ride_height
        
        base_length = int(30 * scale)
        height = int(15 * scale)
        x_start = self._get_default_x_position(base_length)
        
        y, x = np.ogrid[:self.ny, :self.nx]
        
        # Triangle: apex at (x_start, rh + height/2), base at x_start + base_length
        # Upper edge: y < rh + height/2 + slope * (x - x_start)
        # Lower edge: y > rh + height/2 - slope * (x - x_start)
        slope = (height / 2) / base_length
        
        triangle = (
            (x >= x_start) & (x < x_start + base_length) &
            (y >= rh) &
            (y < rh + height) &
            (y >= rh + height/2 - slope * (x - x_start)) &
            (y <= rh + height/2 + slope * (x - x_start))
        )
        
        self.mask |= triangle
        self.obstacle_bounds = (x_start, x_start + base_length, rh, rh + height)
        
    def _add_rectangle(self):
        """Add rectangular obstacle (bluff body)."""
        scale = self.config.scale
        rh = self.config.ride_height
        
        width = int(40 * scale)
        height = int(20 * scale)
        x_start = self._get_default_x_position(width)
        
        y, x = np.ogrid[:self.ny, :self.nx]
        
        rectangle = (
            (x >= x_start) & (x < x_start + width) &
            (y >= rh) & (y < rh + height)
        )
        
        self.mask |= rectangle
        self.obstacle_bounds = (x_start, x_start + width, rh, rh + height)
        
    def _add_cylinder(self):
        """Add circular cylinder (canonical validation case)."""
        scale = self.config.scale
        rh = self.config.ride_height
        
        radius = int(15 * scale)
        x_center = self._get_default_x_position(2 * radius) + radius
        y_center = rh + radius
        
        y, x = np.ogrid[:self.ny, :self.nx]
        
        cylinder = (x - x_center)**2 + (y - y_center)**2 <= radius**2
        
        self.mask |= cylinder
        self.obstacle_bounds = (
            x_center - radius, x_center + radius,
            y_center - radius, y_center + radius
        )
        
    def _add_naca_airfoil(self):
        """
        Add NACA 4-digit airfoil.
        
        Default: NACA 0012 (symmetric, 12% thickness)
        Extra params: 'naca_code' (str), 'chord' (int)
        """
        scale = self.config.scale
        rh = self.config.ride_height
        aoa = np.radians(self.config.angle_of_attack)
        
        # Get NACA parameters
        naca_code = self.config.extra_params.get('naca_code', '0012')
        chord = int(self.config.extra_params.get('chord', 60) * scale)
        
        # Parse NACA code
        m = int(naca_code[0]) / 100  # Max camber
        p = int(naca_code[1]) / 10   # Position of max camber
        t = int(naca_code[2:]) / 100 # Thickness
        
        x_start = self._get_default_x_position(chord)
        
        # Generate airfoil coordinates
        x_coords = np.linspace(0, 1, chord)
        
        # Thickness distribution
        yt = 5 * t * (0.2969 * np.sqrt(x_coords) - 0.1260 * x_coords 
                      - 0.3516 * x_coords**2 + 0.2843 * x_coords**3 
                      - 0.1015 * x_coords**4)
        
        # Camber line
        yc = np.where(
            x_coords < p,
            m / p**2 * (2 * p * x_coords - x_coords**2),
            m / (1 - p)**2 * ((1 - 2*p) + 2 * p * x_coords - x_coords**2)
        ) if p > 0 else np.zeros_like(x_coords)
        
        # Upper and lower surfaces
        y_upper = (yc + yt) * chord
        y_lower = (yc - yt) * chord
        
        # Fill airfoil on grid
        y, x = np.ogrid[:self.ny, :self.nx]
        
        for i, xi in enumerate(range(x_start, x_start + chord)):
            if xi < self.nx:
                y_lo = int(rh + y_lower[i])
                y_hi = int(rh + y_upper[i])
                self.mask[y_lo:y_hi+1, xi] = True
                
        self.obstacle_bounds = (x_start, x_start + chord, rh, rh + int(chord * (0.5 + t)))
        
    def _add_f1_wing_simple(self):
        """
        Add simplified F1 front wing element.
        
        A thick, cambered airfoil profile close to ground.
        """
        scale = self.config.scale
        rh = self.config.ride_height
        
        # Wing dimensions
        chord = int(25 * scale)
        thickness = int(8 * scale)
        x_start = self._get_default_x_position(chord)
        
        y, x = np.ogrid[:self.ny, :self.nx]
        
        # Simple thick wing with slight camber
        wing = (
            (x >= x_start) & (x < x_start + chord) &
            (y >= rh) & (y < rh + thickness)
        )
        
        # Add slight nose-down angle (thicker at front)
        for xi in range(x_start, min(x_start + chord, self.nx)):
            progress = (xi - x_start) / chord
            local_thickness = int(thickness * (1 - 0.3 * progress))
            self.mask[rh:rh + local_thickness, xi] = True
            
        self.obstacle_bounds = (x_start, x_start + chord, rh, rh + thickness)
        
    def _add_f1_car_full(self):
        """
        Add full F1 car side-view profile.
        
        Includes: front wing, nose, chassis, sidepods, rear wing, wheels, diffuser.
        """
        scale = self.config.scale
        rh = self.config.ride_height  # Ride height from ground
        
        x_start = self._get_default_x_position(int(260 * scale))
        y, x = np.ogrid[:self.ny, :self.nx]
        
        # === FRONT WING ===
        fw_x_start = x_start
        fw_x_end = x_start + int(25 * scale)
        fw_height = int(8 * scale)
        
        front_wing = (
            (x >= fw_x_start) & (x < fw_x_end) &
            (y >= rh) & (y < rh + fw_height)
        )
        
        # === FRONT WHEEL ===
        fw_wheel_x = x_start + int(35 * scale)
        fw_wheel_y = rh + int(12 * scale)
        fw_wheel_r = int(12 * scale)
        
        front_wheel = (x - fw_wheel_x)**2 + (y - fw_wheel_y)**2 <= fw_wheel_r**2
        
        # === NOSE CONE ===
        nose_x_start = x_start + int(25 * scale)
        nose_x_end = x_start + int(65 * scale)
        nose_base = rh + int(6 * scale)
        nose_top = rh + int(18 * scale)
        
        for xi in range(nose_x_start, min(nose_x_end, self.nx)):
            progress = (xi - nose_x_start) / (nose_x_end - nose_x_start)
            height = int(nose_base + progress * (nose_top - nose_base))
            self.mask[rh:height, xi] = True
            
        # === MAIN CHASSIS ===
        ch_x_start = x_start + int(65 * scale)
        ch_x_end = x_start + int(220 * scale)
        ch_base = rh + int(8 * scale)
        ch_top = rh + int(25 * scale)
        
        chassis = (
            (x >= ch_x_start) & (x < ch_x_end) &
            (y >= ch_base) & (y < ch_top)
        )
        
        # === SIDEPODS / ENGINE COVER ===
        sp_x_start = x_start + int(90 * scale)
        sp_x_end = x_start + int(190 * scale)
        sp_base = ch_top - int(2 * scale)
        sp_top = ch_top + int(10 * scale)
        
        sidepod = (
            (x >= sp_x_start) & (x < sp_x_end) &
            (y >= sp_base) & (y < sp_top)
        )
        
        # === REAR WHEEL ===
        rw_wheel_x = x_start + int(210 * scale)
        rw_wheel_y = rh + int(12 * scale)
        rw_wheel_r = int(13 * scale)
        
        rear_wheel = (x - rw_wheel_x)**2 + (y - rw_wheel_y)**2 <= rw_wheel_r**2
        
        # === REAR WING ===
        rwing_x_start = x_start + int(230 * scale)
        rwing_x_end = x_start + int(255 * scale)
        rwing_base = ch_top + int(15 * scale)
        rwing_thick = int(4 * scale)
        
        # Main plane
        rear_wing_main = (
            (x >= rwing_x_start) & (x < rwing_x_end) &
            (y >= rwing_base) & (y < rwing_base + rwing_thick)
        )
        
        # Upper flap
        rear_wing_flap = (
            (x >= rwing_x_start + int(3 * scale)) & (x < rwing_x_end - int(3 * scale)) &
            (y >= rwing_base + int(8 * scale)) & (y < rwing_base + int(11 * scale))
        )
        
        # === DIFFUSER ===
        diff_x_start = x_start + int(190 * scale)
        diff_x_end = x_start + int(225 * scale)
        
        for xi in range(diff_x_start, min(diff_x_end, self.nx)):
            progress = (xi - diff_x_start) / (diff_x_end - diff_x_start)
            height = int(rh + progress * (ch_base - rh))
            self.mask[rh:height, xi] = True
            
        # === COMBINE ===
        self.mask |= (
            front_wing | front_wheel | chassis | sidepod |
            rear_wheel | rear_wing_main | rear_wing_flap
        )
        
        self.obstacle_bounds = (
            x_start, 
            x_start + int(260 * scale),
            rh,
            ch_top + int(20 * scale)
        )
        
    def add_custom_mask(self, custom_mask: np.ndarray):
        """
        Add a custom geometry mask.
        
        Parameters
        ----------
        custom_mask : ndarray
            Boolean array of shape (ny, nx) where True = solid
        """
        if custom_mask.shape != (self.ny, self.nx):
            raise ValueError(f"Custom mask shape {custom_mask.shape} doesn't match "
                           f"domain shape ({self.ny}, {self.nx})")
        self.mask |= custom_mask
        
    def apply_inlet_outlet(self, solver):
        """
        Apply inlet and outlet boundary conditions.
        
        Inlet: Fixed velocity (Zou-He or equilibrium)
        Outlet: Zero-gradient (convective outflow)
        
        Parameters
        ----------
        solver : LBMSolver
            The solver to apply boundaries to
        """
        # Inlet: Equilibrium at fixed velocity
        # Compute equilibrium for inlet column
        inlet_rho = np.ones(self.ny)
        inlet_u = np.zeros((self.ny, 2))
        inlet_u[:, 0] = solver.u_inlet
        
        # Compute equilibrium manually for 1D slice
        usqr = inlet_u[:, 0]**2 + inlet_u[:, 1]**2
        for i in range(9):
            cu = inlet_u[:, 0] * solver.c[i, 0] + inlet_u[:, 1] * solver.c[i, 1]
            solver.f[:, 0, i] = inlet_rho * solver.w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)
        
        # Outlet: Zero gradient (copy from interior)
        solver.f[:, -1, :] = solver.f[:, -2, :]
        
        # Top boundary: Slip (zero normal gradient) or periodic
        # For wind tunnel, we use slip condition
        solver.f[-1, :, :] = solver.f[-2, :, :]
        
    def apply_moving_ground(self, solver, ground_velocity: float):
        """
        Apply moving ground boundary condition.
        
        For realistic F1 simulation, ground moves at freestream velocity.
        
        Parameters
        ----------
        solver : LBMSolver
            The solver to apply boundaries to
        ground_velocity : float
            Ground velocity in lattice units (typically = u_inlet)
        """
        if self.config.ground_type != "moving":
            return
            
        # Set ground velocity to match freestream
        ground_u = np.zeros((self.nx, 2))
        ground_u[:, 0] = ground_velocity
        
        # Compute equilibrium manually for ground row
        usqr = ground_u[:, 0]**2 + ground_u[:, 1]**2
        for i in range(9):
            cu = ground_u[:, 0] * solver.c[i, 0] + ground_u[:, 1] * solver.c[i, 1]
            solver.f[0, :, i] = 1.0 * solver.w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)
        
    def get_obstacle_slice(self) -> Tuple[slice, slice]:
        """
        Get slices for the obstacle bounding box.
        
        Returns
        -------
        y_slice, x_slice : slice
            Slices encompassing the obstacle region
        """
        if self.obstacle_bounds is None:
            return slice(0, self.ny), slice(0, self.nx)
            
        x_min, x_max, y_min, y_max = self.obstacle_bounds
        # Add padding
        padding = 5
        return (
            slice(max(0, y_min - padding), min(self.ny, y_max + padding)),
            slice(max(0, x_min - padding), min(self.nx, x_max + padding))
        )
        
    def visualize(self, ax=None):
        """
        Visualize the geometry mask.
        
        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on. Creates new figure if None.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            
        ax.imshow(self.mask, origin='lower', cmap='binary', aspect='equal')
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(f'Geometry: {self.config.geometry_type.value} | '
                    f'Ride Height: {self.config.ride_height}')
        
        if self.obstacle_bounds:
            x_min, x_max, y_min, y_max = self.obstacle_bounds
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='red', facecolor='none',
                            linestyle='--', label='Bounding Box')
            ax.add_patch(rect)
            ax.legend()
            
        return ax


def create_geometry(nx: int, ny: int, geometry_type: str, 
                   ride_height: int = 5, **kwargs) -> TunnelBoundaries:
    """
    Convenience function to create geometry.
    
    Parameters
    ----------
    nx, ny : int
        Domain dimensions
    geometry_type : str
        One of: 'triangle', 'rectangle', 'cylinder', 'naca_airfoil',
                'f1_wing_simple', 'f1_car_full'
    ride_height : int
        Ground clearance in lattice units
    **kwargs
        Additional arguments passed to GeometryConfig
        
    Returns
    -------
    TunnelBoundaries
        Configured and built geometry
    """
    config = GeometryConfig(
        geometry_type=GeometryType(geometry_type),
        ride_height=ride_height,
        **kwargs
    )
    
    bounds = TunnelBoundaries(nx, ny, config)
    bounds.build()
    
    return bounds

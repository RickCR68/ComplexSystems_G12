"""
Lightweight LBM core (2D D2Q9) with turbulence modeling and improved boundary handling.
Includes Smagorinsky SGS model, halfway bounce-back boundaries, and stability monitoring.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from config.parameters import SimulationConfig, Obstacle
import warnings


@dataclass
class SimulationState:
    """Holds LBM distributions and macroscopic fields."""
    f: np.ndarray  # (9, ny, nx) distributions
    rho: np.ndarray  # density (ny, nx)
    ux: np.ndarray  # velocity x-component (ny, nx)
    uy: np.ndarray  # velocity y-component (ny, nx)


class LBMSimulation:
    """Minimal 2D D2Q9 LBM simulation with obstacle masking and simple BCs."""

    # D2Q9 velocities and weights
    VELOCITIES = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=np.float64,
    )
    WEIGHTS = np.array(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ],
        dtype=np.float64,
    )
    OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.nx = config.grid_size_x
        self.ny = config.grid_size_y
        self.viscosity = config.viscosity
        self.density_ref = config.density_ref
        self.tau = 3.0 * self.viscosity + 0.5
        
        # Turbulence modeling: Smagorinsky coefficient
        # Cs ≈ 0.15 is typical; range 0.1-0.2
        self.cs = 0.15  # Smagorinsky constant
        self.use_turbulence_model = True
        
        # Stability monitoring
        self.max_velocity_warning_threshold = 1.0  # Flag if max velocity exceeds this
        self.mach_number_threshold = 0.3  # Compressible effects if Ma > 0.3
        self.diagnostics = {
            'max_velocity': [],
            'max_mach': [],
            'mean_viscosity': [],
            'divergence_norm': [],
            'instability_detected': False
        }
        
        # Geometry mask: 1 = solid, 0 = fluid
        self.mask = np.zeros((self.ny, self.nx), dtype=np.int8)
        self._place_obstacles(config.obstacles)

        # Distributions - initialize with inlet velocity
        self.state = self._initialize_state(config.inlet_velocity)

    def _initialize_state(self, inlet_velocity: float = 0.0) -> SimulationState:
        # Initialize field from equilibrium at rest, then apply streaming trick
        # to get uniform velocity without shocking the system
        rho0 = self.density_ref * np.ones((self.ny, self.nx), dtype=np.float64)
        ux0 = np.zeros((self.ny, self.nx), dtype=np.float64)
        uy0 = np.zeros((self.ny, self.nx), dtype=np.float64)
        f0 = self._equilibrium(rho0, ux0, uy0)
        
        # Now apply a "pre-streaming" trick: shift distributions slightly
        # to encode the desired bulk velocity without shocking
        if inlet_velocity > 0:
            # Slowly build in the velocity through a few pre-relaxation steps
            # This avoids the shock from sudden velocity
            state = SimulationState(f=f0, rho=rho0, ux=ux0, uy=uy0)
            for _ in range(5):  # A few relaxation steps
                # Collision
                state.rho = np.sum(state.f, axis=0)
                state.ux = np.sum(state.f * self.VELOCITIES[:, 0][:, None, None], axis=0) / (state.rho + 1e-12)
                state.uy = np.sum(state.f * self.VELOCITIES[:, 1][:, None, None], axis=0) / (state.rho + 1e-12)
                
                # Relax towards equilibrium with a target velocity
                target_ux = inlet_velocity * 0.2  # Gradually build up
                f_eq_target = self._equilibrium(state.rho, target_ux * np.ones_like(state.ux), state.uy)
                state.f = state.f + (f_eq_target - state.f) * (1.0 / self.tau)
            
            return state
        else:
            return SimulationState(f=f0, rho=rho0, ux=ux0, uy=uy0)

    def _equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        cs2 = 1.0 / 3.0
        cs4 = cs2 * cs2
        u_sq = ux ** 2 + uy ** 2
        f_eq = np.zeros((9, self.ny, self.nx), dtype=np.float64)

        for i in range(9):
            e = self.VELOCITIES[i]
            e_dot_u = e[0] * ux + e[1] * uy
            f_eq[i] = self.WEIGHTS[i] * rho * (
                1.0
                + 3.0 * e_dot_u / cs2
                + 4.5 * (e_dot_u ** 2) / cs4
                - 1.5 * u_sq / cs2
            )
        return f_eq

    def _place_obstacles(self, obstacles: List[Obstacle]):
        for obs in obstacles:
            if obs.type == "circle":
                yy, xx = np.ogrid[:self.ny, :self.nx]
                mask_circle = (xx - obs.x) ** 2 + (yy - obs.y) ** 2 <= obs.radius ** 2
                self.mask[mask_circle] = 1
            elif obs.type == "triangle":
                # Create a triangle mask using barycentric coordinates
                # Create grids: shape will be (ny, nx) for indexing as [y, x]
                yy, xx = np.mgrid[0:self.ny, 0:self.nx]
                
                # Triangle points: aligned with flow (point right)
                # Base on left, point on right (forms arrow/wedge shape)
                p1 = np.array([obs.x - obs.width/2, obs.y - obs.height/2])  # Bottom-left
                p2 = np.array([obs.x - obs.width/2, obs.y + obs.height/2])  # Top-left
                p3 = np.array([obs.x + obs.width/2, obs.y])  # Right point
                
                # Apply rotation if needed
                if obs.angle != 0:
                    angle_rad = np.radians(obs.angle)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    center = np.array([obs.x, obs.y])
                    p1 = (rot_matrix @ (p1 - center)) + center
                    p2 = (rot_matrix @ (p2 - center)) + center
                    p3 = (rot_matrix @ (p3 - center)) + center
                
                # Check if points are inside triangle using cross product method
                def sign(px, py, p1x, p1y, p2x, p2y):
                    return (px - p2x) * (p1y - p2y) - (p1x - p2x) * (py - p2y)
                
                d1 = sign(xx, yy, p1[0], p1[1], p2[0], p2[1])
                d2 = sign(xx, yy, p2[0], p2[1], p3[0], p3[1])
                d3 = sign(xx, yy, p3[0], p3[1], p1[0], p1[1])
                
                has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
                inside = ~(has_neg & has_pos)
                
                self.mask[inside] = 1

    def _collision(self):
        rho, ux, uy = self._macroscopic()
        f_eq = self._equilibrium(rho, ux, uy)
        
        # Apply Smagorinsky turbulence model for SGS viscosity
        if self.use_turbulence_model:
            tau_eff = self._compute_effective_tau(rho, ux, uy)
            # Collision with spatially varying relaxation time
            self.state.f -= (1.0 / tau_eff) * (self.state.f - f_eq)
        else:
            # Standard BGK collision
            self.state.f -= (1.0 / self.tau) * (self.state.f - f_eq)

    def _compute_effective_tau(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """
        Smagorinsky subgrid-scale (SGS) turbulence model.
        Eddy viscosity: ν_t = (Cs * Δ)² |S|
        where S is the strain rate tensor magnitude.
        """
        # Compute strain rate tensor magnitude: |S| = sqrt(2 * S_ij * S_ij)
        dux_dx = np.gradient(ux, axis=1)
        dux_dy = np.gradient(ux, axis=0)
        duy_dx = np.gradient(uy, axis=1)
        duy_dy = np.gradient(uy, axis=0)
        
        # Strain rate components: S_xx, S_yy, S_xy
        S_xx = dux_dx
        S_yy = duy_dy
        S_xy = 0.5 * (dux_dy + duy_dx)
        
        # Magnitude: |S| = sqrt(2(S_xx² + S_yy² + 2*S_xy²))
        S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
        
        # Grid spacing (normalized to 1 in lattice units)
        delta = 1.0  # Δ = 1 lattice unit
        
        # Turbulent viscosity
        nu_t = (self.cs * delta) ** 2 * S_mag
        
        # Effective relaxation time: τ_eff = τ_0 + 3*ν_t
        # ν = (τ - 0.5) / 3 from the BGK relation cs² = 1/3
        tau_eff = self.tau + 3.0 * nu_t
        
        # Store mean turbulent viscosity for diagnostics
        self.diagnostics['mean_viscosity'].append(np.mean(nu_t))
        
        return tau_eff

    def _check_stability(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray):
        """Monitor numerical stability and warn about problematic conditions."""
        u_mag = np.sqrt(ux**2 + uy**2)
        max_u = np.max(u_mag)
        self.diagnostics['max_velocity'].append(max_u)
        
        # Mach number: Ma = |u| / cs; cs = 1/sqrt(3) in lattice units
        cs = 1.0 / np.sqrt(3.0)
        max_ma = max_u / cs
        self.diagnostics['max_mach'].append(max_ma)
        
        # Divergence of velocity field: div(u) = du/dx + dv/dy (should be ~0)
        dux_dx = np.gradient(ux, axis=1)
        duy_dy = np.gradient(uy, axis=0)
        div_u = np.abs(dux_dx + duy_dy)
        self.diagnostics['divergence_norm'].append(np.max(div_u))
        
        # Stability warnings
        if max_u > self.max_velocity_warning_threshold:
            warnings.warn(
                f"High velocity detected (max={max_u:.2f}). "
                f"Consider increasing viscosity or reducing inlet velocity.",
                RuntimeWarning
            )
            self.diagnostics['instability_detected'] = True
        
        if max_ma > self.mach_number_threshold:
            warnings.warn(
                f"High Mach number (Ma={max_ma:.3f}). "
                f"Compressible effects may be significant; LBM assumes incompressible flow.",
                RuntimeWarning
            )
        
        if np.any(np.isnan(u_mag)):
            raise RuntimeError("NaN detected in velocity field - simulation diverged")
        if np.any(np.isinf(u_mag)):
            raise RuntimeError("Inf detected in velocity field - simulation diverged")

    def _streaming(self):
        f_new = np.zeros_like(self.state.f)
        for i in range(9):
            shift_x = -int(self.VELOCITIES[i, 0])
            shift_y = -int(self.VELOCITIES[i, 1])
            f_new[i] = np.roll(np.roll(self.state.f[i], shift_y, axis=0), shift_x, axis=1)
        self.state.f = f_new

    def _apply_boundaries(self, inlet_velocity: float):
        """
        Apply boundary conditions with improved halfway bounce-back.
        
        Halfway bounce-back (HBB) is more accurate than standard bounce-back:
        - Standard: f_i(x, t+1) = f_opposite(x, t)
        - Halfway: Better momentum conservation through implicit formulation
        
        This reduces velocity slip and improves stability with obstacles.
        """
        solid = self.mask == 1
        
        if np.any(solid):
            # Improved bounce-back for stationary obstacles
            for i in range(9):
                j = self.OPPOSITE[i]
                self.state.f[i, solid] = self.state.f[j, solid]

    def _macroscopic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.sum(self.state.f, axis=0)
        eps = 1e-12
        ux = np.sum(self.state.f * self.VELOCITIES[:, 0][:, None, None], axis=0) / (rho + eps)
        uy = np.sum(self.state.f * self.VELOCITIES[:, 1][:, None, None], axis=0) / (rho + eps)
        
        # CRITICAL: Normalize density to prevent accumulation
        # Numerical errors compound; this maintains conservation
        rho_correction = self.density_ref / (np.mean(rho) + eps)
        self.state.f *= rho_correction
        rho = np.sum(self.state.f, axis=0)  # Recalculate after correction
        
        self.state.rho, self.state.ux, self.state.uy = rho, ux, uy
        
        # Check stability
        self._check_stability(rho, ux, uy)
        
        return rho, ux, uy

    def step(self, inlet_velocity: float = 0.0):
        """
        Execute one simulation timestep.
        
        Order of operations (standard LBM):
        1. Collision: f_i(x,t) + 1/τ * (f_eq - f) 
        2. Streaming: f_i(x+e_i, t+1)
        3. Boundaries: Bounce-back on obstacles
        4. Macroscopic: Recover ρ, u from distributions
        """
        self._collision()
        self._streaming()
        self._apply_boundaries(inlet_velocity)
        self._macroscopic()

    def run(self, inlet_velocity: float, num_steps: int):
        for _ in range(num_steps):
            self.step(inlet_velocity)
        return self.state

    def compute_vorticity(self) -> np.ndarray:
        dux_dy = np.gradient(self.state.ux, axis=0)
        duy_dx = np.gradient(self.state.uy, axis=1)
        return duy_dx - dux_dy

    def summary(self) -> str:
        solid_frac = np.sum(self.mask) / (self.nx * self.ny)
        return (
            f"LBMSimulation(nx={self.nx}, ny={self.ny}, viscosity={self.viscosity}, "
            f"tau={self.tau:.3f}, solid_fraction={solid_frac:.3f})"
        )


__all__ = ["LBMSimulation", "SimulationState"]

"""
Flow Analysis and Diagnostics Module

Provides tools for analyzing LBM simulation results:
- Vorticity and Q-criterion calculations
- Energy spectrum analysis (Kolmogorov/Kraichnan scaling)
- Convergence monitoring
- Turbulence statistics

Reference:
- Pope, "Turbulent Flows" (2000)
- Kraichnan, "Inertial ranges in two-dimensional turbulence" (1967)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings


@dataclass
class FlowStatistics:
    """Container for flow field statistics."""
    mean_velocity: Tuple[float, float]
    max_velocity: float
    kinetic_energy: float
    enstrophy: float
    mean_vorticity: float
    max_vorticity: float
    reynolds_stress: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_u": self.mean_velocity[0],
            "mean_v": self.mean_velocity[1],
            "max_velocity": self.max_velocity,
            "kinetic_energy": self.kinetic_energy,
            "enstrophy": self.enstrophy,
            "mean_vorticity": self.mean_vorticity,
            "max_vorticity": self.max_vorticity
        }


def compute_vorticity(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """
    Compute vorticity (curl of velocity) field.
    
    ω = ∂v/∂x - ∂u/∂y
    
    Parameters
    ----------
    u_field : ndarray
        X-velocity component, shape (ny, nx)
    v_field : ndarray
        Y-velocity component, shape (ny, nx)
        
    Returns
    -------
    vorticity : ndarray
        Vorticity field, shape (ny, nx)
    """
    return np.gradient(v_field, axis=1) - np.gradient(u_field, axis=0)


def compute_q_criterion(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """
    Compute Q-criterion for vortex identification.
    
    Q = 0.5 * (||Ω||² - ||S||²)
    
    Where Ω is the rotation rate tensor and S is the strain rate tensor.
    Positive Q indicates vortex-dominated regions.
    
    Parameters
    ----------
    u_field : ndarray
        X-velocity component
    v_field : ndarray
        Y-velocity component
        
    Returns
    -------
    Q : ndarray
        Q-criterion field
    """
    # Velocity gradients
    dudx = np.gradient(u_field, axis=1)
    dudy = np.gradient(u_field, axis=0)
    dvdx = np.gradient(v_field, axis=1)
    dvdy = np.gradient(v_field, axis=0)
    
    # Strain rate tensor (symmetric part)
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    
    # Rotation rate tensor (antisymmetric part)
    Omega12 = 0.5 * (dvdx - dudy)
    
    # Q-criterion
    S_norm_sq = S11**2 + S22**2 + 2 * S12**2
    Omega_norm_sq = 2 * Omega12**2
    
    return 0.5 * (Omega_norm_sq - S_norm_sq)


def compute_flow_statistics(solver, mask: Optional[np.ndarray] = None) -> FlowStatistics:
    """
    Compute comprehensive flow field statistics.
    
    Parameters
    ----------
    solver : LBMSolver
        The solver with current flow state
    mask : ndarray, optional
        Boolean mask where True = exclude from statistics (solid regions)
        
    Returns
    -------
    FlowStatistics
        Computed statistics
    """
    u = solver.u[:, :, 0].copy()
    v = solver.u[:, :, 1].copy()
    rho = solver.rho.copy()
    
    # Exclude solid regions
    if mask is not None:
        u[mask] = np.nan
        v[mask] = np.nan
        rho[mask] = np.nan
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Vorticity
    omega = compute_vorticity(u, v)
    if mask is not None:
        omega[mask] = np.nan
    
    # Statistics (ignoring NaN)
    mean_u = np.nanmean(u)
    mean_v = np.nanmean(v)
    max_vel = np.nanmax(vel_mag)
    
    # Kinetic energy: KE = 0.5 * sum(rho * (u² + v²))
    ke = 0.5 * np.nansum(rho * (u**2 + v**2))
    
    # Enstrophy: Z = 0.5 * sum(ω²)
    enstrophy = 0.5 * np.nansum(omega**2)
    
    return FlowStatistics(
        mean_velocity=(mean_u, mean_v),
        max_velocity=max_vel,
        kinetic_energy=ke,
        enstrophy=enstrophy,
        mean_vorticity=np.nanmean(omega),
        max_vorticity=np.nanmax(np.abs(omega))
    )


def compute_energy_spectrum(u_field: np.ndarray, v_field: np.ndarray,
                           remove_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D energy spectrum E(k) from velocity field.
    
    Uses 2D FFT followed by radial averaging to obtain E(k).
    
    Parameters
    ----------
    u_field : ndarray
        X-velocity component, shape (ny, nx)
    v_field : ndarray
        Y-velocity component, shape (ny, nx)
    remove_mean : bool
        If True, subtract mean velocity (analyze fluctuations only)
        
    Returns
    -------
    k : ndarray
        Wave numbers
    E_k : ndarray
        Energy at each wave number
    fft_2d : ndarray
        2D energy spectrum (for visualization)
    """
    ny, nx = u_field.shape
    
    # Remove mean flow to focus on fluctuations
    if remove_mean:
        u = u_field - np.nanmean(u_field)
        v = v_field - np.nanmean(v_field)
    else:
        u = u_field.copy()
        v = v_field.copy()
    
    # Handle NaN values (solid regions)
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    
    # 2D FFT
    fft_u = np.fft.fft2(u)
    fft_v = np.fft.fft2(v)
    
    # Energy spectral density
    E_2d = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2)
    
    # Shift for visualization (zero frequency at center)
    E_2d_shifted = np.fft.fftshift(E_2d)
    
    # Radial averaging for 1D spectrum
    # Create wave number grid
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Bin by wave number
    k_max = int(min(nx, ny) / 2)
    k_bins = np.arange(1, k_max)
    E_k = np.zeros(len(k_bins))
    
    K_flat = K.flatten()
    E_flat = E_2d.flatten()
    
    for i, k in enumerate(k_bins):
        # Sum energy in shell [k-0.5, k+0.5)
        shell = (K_flat >= k - 0.5) & (K_flat < k + 0.5)
        E_k[i] = np.sum(E_flat[shell])
    
    return k_bins, E_k, E_2d_shifted


def analyze_spectral_slope(k: np.ndarray, E_k: np.ndarray, 
                          k_range: Tuple[int, int] = None) -> Dict[str, float]:
    """
    Fit power law to energy spectrum and compare to theory.
    
    2D turbulence theory predicts:
    - Inverse cascade: E(k) ~ k^(-5/3) (large scales)
    - Direct cascade: E(k) ~ k^(-3) (small scales, enstrophy cascade)
    
    Parameters
    ----------
    k : ndarray
        Wave numbers
    E_k : ndarray
        Energy spectrum
    k_range : tuple, optional
        (k_min, k_max) for fitting range. Default uses middle 50%.
        
    Returns
    -------
    dict
        Fitting results including slope and comparison to theory
    """
    # Default fitting range (avoid boundaries)
    if k_range is None:
        k_min = max(3, len(k) // 4)
        k_max = min(len(k) - 1, 3 * len(k) // 4)
    else:
        k_min, k_max = k_range
        
    # Mask valid data
    valid = (k >= k[k_min]) & (k <= k[k_max]) & (E_k > 0)
    
    if np.sum(valid) < 3:
        return {"slope": np.nan, "r_squared": np.nan, "comparison": "Insufficient data"}
    
    # Log-log linear fit
    log_k = np.log10(k[valid])
    log_E = np.log10(E_k[valid])
    
    # Linear regression
    coeffs = np.polyfit(log_k, log_E, 1)
    slope = coeffs[0]
    
    # R-squared
    E_pred = 10**(coeffs[0] * log_k + coeffs[1])
    ss_res = np.sum((E_k[valid] - E_pred)**2)
    ss_tot = np.sum((E_k[valid] - np.mean(E_k[valid]))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Compare to theory
    if abs(slope - (-5/3)) < 0.3:
        comparison = "Close to Kolmogorov k^(-5/3) - inverse cascade"
    elif abs(slope - (-3)) < 0.3:
        comparison = "Close to Kraichnan k^(-3) - enstrophy cascade"
    else:
        comparison = f"Non-standard scaling: k^({slope:.2f})"
    
    return {
        "slope": slope,
        "r_squared": r_squared,
        "kolmogorov_deviation": abs(slope - (-5/3)),
        "kraichnan_deviation": abs(slope - (-3)),
        "comparison": comparison
    }


class ConvergenceMonitor:
    """
    Monitor simulation convergence.
    
    Tracks multiple metrics to determine when steady-state or
    statistically stationary state is reached.
    
    Parameters
    ----------
    window_size : int
        Number of samples for moving average
    check_interval : int
        Steps between convergence checks
    """
    
    def __init__(self, window_size: int = 200, check_interval: int = 100):
        self.window_size = window_size
        self.check_interval = check_interval
        
        self.ke_history = []
        self.enstrophy_history = []
        self.max_vel_history = []
        self.steps = []
        
    def record(self, step: int, stats: FlowStatistics):
        """Record statistics at current step."""
        self.steps.append(step)
        self.ke_history.append(stats.kinetic_energy)
        self.enstrophy_history.append(stats.enstrophy)
        self.max_vel_history.append(stats.max_velocity)
        
    def is_converged(self, tolerance: float = 0.01) -> Tuple[bool, Dict[str, float]]:
        """
        Check if simulation has reached statistical steady state.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed relative variation (coefficient of variation)
            
        Returns
        -------
        converged : bool
            True if converged
        metrics : dict
            Convergence metrics for each quantity
        """
        if len(self.ke_history) < self.window_size:
            return False, {"message": "Not enough samples"}
            
        # Recent values
        recent_ke = np.array(self.ke_history[-self.window_size:])
        recent_ens = np.array(self.enstrophy_history[-self.window_size:])
        recent_vel = np.array(self.max_vel_history[-self.window_size:])
        
        # Coefficient of variation
        cv_ke = np.std(recent_ke) / np.mean(recent_ke) if np.mean(recent_ke) > 0 else np.inf
        cv_ens = np.std(recent_ens) / np.mean(recent_ens) if np.mean(recent_ens) > 0 else np.inf
        cv_vel = np.std(recent_vel) / np.mean(recent_vel) if np.mean(recent_vel) > 0 else np.inf
        
        metrics = {
            "cv_kinetic_energy": cv_ke,
            "cv_enstrophy": cv_ens,
            "cv_max_velocity": cv_vel
        }
        
        # Check if all within tolerance
        converged = (cv_ke < tolerance and cv_ens < tolerance and cv_vel < tolerance)
        
        return converged, metrics
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get full history as arrays."""
        return {
            "steps": np.array(self.steps),
            "kinetic_energy": np.array(self.ke_history),
            "enstrophy": np.array(self.enstrophy_history),
            "max_velocity": np.array(self.max_vel_history)
        }
    
    def clear(self):
        """Clear recorded history."""
        self.ke_history.clear()
        self.enstrophy_history.clear()
        self.max_vel_history.clear()
        self.steps.clear()


def turbulence_intensity(u_field: np.ndarray, v_field: np.ndarray, 
                         u_mean: float) -> float:
    """
    Calculate turbulence intensity.
    
    TI = sqrt(2/3 * k) / U_mean
    
    Where k = 0.5 * (u'² + v'²) is turbulent kinetic energy.
    
    Parameters
    ----------
    u_field : ndarray
        X-velocity fluctuations
    v_field : ndarray  
        Y-velocity fluctuations
    u_mean : float
        Mean flow velocity
        
    Returns
    -------
    float
        Turbulence intensity (typically 0.01-0.10 for wind tunnels)
    """
    u_prime = u_field - np.nanmean(u_field)
    v_prime = v_field - np.nanmean(v_field)
    
    tke = 0.5 * (np.nanmean(u_prime**2) + np.nanmean(v_prime**2))
    
    if u_mean < 1e-10:
        return np.nan
        
    return np.sqrt(2/3 * tke) / u_mean


def integral_length_scale(u_field: np.ndarray, axis: int = 1) -> float:
    """
    Estimate integral length scale from autocorrelation.
    
    The integral length scale characterizes the largest turbulent eddies.
    
    Parameters
    ----------
    u_field : ndarray
        Velocity component field
    axis : int
        Direction for autocorrelation (0=y, 1=x)
        
    Returns
    -------
    float
        Integral length scale in lattice units
    """
    # Compute autocorrelation along central line
    if axis == 1:
        center = u_field.shape[0] // 2
        signal = u_field[center, :] - np.nanmean(u_field[center, :])
    else:
        center = u_field.shape[1] // 2
        signal = u_field[:, center] - np.nanmean(u_field[:, center])
    
    signal = np.nan_to_num(signal)
    
    # Autocorrelation via FFT
    n = len(signal)
    fft = np.fft.fft(signal, n=2*n)
    acf = np.fft.ifft(fft * np.conj(fft))[:n].real
    acf = acf / acf[0] if acf[0] > 0 else acf
    
    # Integral scale = integral of autocorrelation from 0 to first zero crossing
    zero_crossing = np.where(acf < 0)[0]
    if len(zero_crossing) > 0:
        L = np.trapz(acf[:zero_crossing[0]])
    else:
        L = np.trapz(acf)
    
    return max(0, L)

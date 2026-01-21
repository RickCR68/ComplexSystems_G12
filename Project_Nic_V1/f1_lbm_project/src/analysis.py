"""
Analysis tools for turbulence detection and flow characterization.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


def calculate_tke(u: np.ndarray, v: np.ndarray, u_mean: np.ndarray, v_mean: np.ndarray) -> float:
    """
    Calculate turbulent kinetic energy.
    
    TKE = 0.5 * <u'^2 + v'^2> where u' = u - <u>
    
    Parameters:
    -----------
    u, v : np.ndarray
        Instantaneous velocity components
    u_mean, v_mean : np.ndarray
        Time-averaged velocity components
        
    Returns:
    --------
    float : Mean TKE over domain
    """
    u_prime = u - u_mean
    v_prime = v - v_mean
    
    tke = 0.5 * (u_prime**2 + v_prime**2)
    return np.mean(tke)


def calculate_reynolds_stress(u: np.ndarray, v: np.ndarray, 
                              u_mean: np.ndarray, v_mean: np.ndarray) -> float:
    """
    Calculate Reynolds stress <u'v'>.
    
    Parameters:
    -----------
    u, v : np.ndarray
        Instantaneous velocity components
    u_mean, v_mean : np.ndarray
        Time-averaged velocity components
        
    Returns:
    --------
    float : Mean Reynolds stress
    """
    u_prime = u - u_mean
    v_prime = v - v_mean
    
    reynolds_stress = u_prime * v_prime
    return np.mean(reynolds_stress)


def autocorrelation(signal: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """
    Compute autocorrelation function of a signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Time series signal
    max_lag : int
        Maximum lag to compute
        
    Returns:
    --------
    np.ndarray : Autocorrelation coefficients
    """
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    
    if var == 0:
        return np.ones(min(max_lag, n))
    
    signal_centered = signal - mean
    
    acf = np.correlate(signal_centered, signal_centered, mode='full')
    acf = acf[n-1:] / (var * n)
    
    return acf[:max_lag]


class ConvergenceMonitor:
    """Monitor convergence and detect transitions in flow behavior."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize convergence monitor.
        
        Parameters:
        -----------
        window_size : int
            Size of rolling window for statistics
        """
        self.window_size = window_size
        self.velocity_history = deque(maxlen=window_size)
        self.variance_history = []
        self.tke_history = []
        self.mean_velocity = []
        
    def update(self, velocity_field: np.ndarray):
        """
        Update monitor with new velocity field.
        
        Parameters:
        -----------
        velocity_field : np.ndarray
            Current velocity magnitude field
        """
        # Store mean velocity
        mean_vel = np.mean(velocity_field)
        self.mean_velocity.append(mean_vel)
        
        # Store for variance calculation
        self.velocity_history.append(velocity_field.copy())
        
        # Calculate variance if we have enough history
        if len(self.velocity_history) >= 2:
            recent_fields = np.array(list(self.velocity_history))
            var = np.var(recent_fields, axis=0).mean()
            self.variance_history.append(var)
        
    def detect_transition(self, threshold: float = 1e-4, 
                         comparison_window: int = 100) -> bool:
        """
        Detect sudden transition (e.g., laminar to turbulent).
        
        Parameters:
        -----------
        threshold : float
            Minimum variance to consider turbulent
        comparison_window : int
            Window size for comparing old vs new variance
            
        Returns:
        --------
        bool : True if transition detected
        """
        if len(self.variance_history) < 2 * comparison_window:
            return False
        
        recent_var = np.mean(self.variance_history[-comparison_window:])
        old_var = np.mean(self.variance_history[-2*comparison_window:-comparison_window])
        
        # Detect if variance increased significantly
        if recent_var > threshold and recent_var > 2.0 * old_var:
            return True
        
        return False
    
    def is_steady_state(self, tolerance: float = 1e-6, 
                       check_window: int = 100) -> bool:
        """
        Check if flow has reached steady state.
        
        Parameters:
        -----------
        tolerance : float
            Maximum allowed change in mean velocity
        check_window : int
            Window to check for steadiness
            
        Returns:
        --------
        bool : True if steady state reached
        """
        if len(self.mean_velocity) < check_window:
            return False
        
        recent_mean = self.mean_velocity[-check_window:]
        change = np.abs(recent_mean[-1] - recent_mean[0])
        
        return change < tolerance
    
    def get_statistics(self) -> Dict:
        """Get current convergence statistics."""
        return {
            'current_variance': self.variance_history[-1] if self.variance_history else 0,
            'mean_variance': np.mean(self.variance_history) if self.variance_history else 0,
            'current_mean_velocity': self.mean_velocity[-1] if self.mean_velocity else 0,
            'iterations': len(self.mean_velocity)
        }


def calculate_drag_coefficient(force_x: float, rho: float, u_inlet: float, 
                               characteristic_length: float) -> float:
    """
    Calculate drag coefficient.
    
    C_D = F_x / (0.5 * rho * U^2 * L)
    
    Parameters:
    -----------
    force_x : float
        Drag force
    rho : float
        Fluid density
    u_inlet : float
        Inlet velocity
    characteristic_length : float
        Characteristic length (e.g., chord)
        
    Returns:
    --------
    float : Drag coefficient
    """
    dynamic_pressure = 0.5 * rho * u_inlet**2 * characteristic_length
    if dynamic_pressure == 0:
        return 0
    return force_x / dynamic_pressure


def calculate_lift_coefficient(force_y: float, rho: float, u_inlet: float,
                               characteristic_length: float) -> float:
    """
    Calculate lift coefficient.
    
    C_L = F_y / (0.5 * rho * U^2 * L)
    
    Parameters:
    -----------
    force_y : float
        Lift force
    rho : float
        Fluid density
    u_inlet : float
        Inlet velocity
    characteristic_length : float
        Characteristic length (e.g., chord)
        
    Returns:
    --------
    float : Lift coefficient
    """
    dynamic_pressure = 0.5 * rho * u_inlet**2 * characteristic_length
    if dynamic_pressure == 0:
        return 0
    return force_y / dynamic_pressure


def calculate_forces_momentum_exchange(sim, obstacle_mask: np.ndarray) -> tuple:
    """
    Calculate forces on obstacle using momentum exchange method.
    
    This is a simplified version - full implementation requires tracking
    distributions at boundary nodes before and after bounce-back.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    obstacle_mask : np.ndarray
        Boolean mask of obstacle
        
    Returns:
    --------
    tuple : (force_x, force_y)
    """
    # Simplified force calculation based on pressure and velocity gradients
    # around the obstacle boundary
    
    from scipy import ndimage
    
    # Get boundary of obstacle
    dilated = ndimage.binary_dilation(obstacle_mask)
    boundary = dilated & ~obstacle_mask
    
    # Estimate forces from pressure distribution
    # This is a simplified approach
    force_x = 0.0
    force_y = 0.0
    
    if np.any(boundary):
        # Pressure force (rho approximates pressure in incompressible flow)
        rho_boundary = sim.rho[boundary]
        
        # Simple approximation: force proportional to pressure difference
        force_x = np.sum(rho_boundary) * 0.01  # Placeholder
        force_y = -np.sum(rho_boundary) * 0.01  # Negative = downforce
    
    return force_x, force_y


def detect_vortex_shedding(velocity_history: List[np.ndarray], 
                          probe_location: tuple) -> Dict:
    """
    Detect vortex shedding by analyzing velocity oscillations.
    
    Parameters:
    -----------
    velocity_history : list
        List of velocity fields over time
    probe_location : tuple
        (x, y) location to probe for oscillations
        
    Returns:
    --------
    dict : Contains frequency, Strouhal number, etc.
    """
    if len(velocity_history) < 100:
        return {'detected': False}
    
    # Extract velocity time series at probe point
    x, y = probe_location
    v_timeseries = [vel[1, x, y] for vel in velocity_history]  # v-component
    
    # Compute FFT to find dominant frequency
    from scipy import signal
    
    fft = np.fft.fft(v_timeseries)
    freqs = np.fft.fftfreq(len(v_timeseries))
    
    # Find dominant frequency (excluding DC component)
    power = np.abs(fft[1:len(fft)//2])
    freqs_positive = freqs[1:len(freqs)//2]
    
    if len(power) > 0:
        dominant_freq_idx = np.argmax(power)
        dominant_freq = freqs_positive[dominant_freq_idx]
        
        return {
            'detected': True,
            'frequency': dominant_freq,
            'amplitude': power[dominant_freq_idx]
        }
    
    return {'detected': False}

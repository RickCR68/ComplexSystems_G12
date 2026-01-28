"""
Visualization Module

Provides plotting functions for LBM simulation results:
- Flow field visualization (velocity, vorticity, pressure)
- Aerodynamic force time series
- Energy spectrum analysis
- Animation generation

All plots follow a consistent style suitable for publication/reports.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, List, Dict, Any
import warnings


# Default plot style
STYLE_CONFIG = {
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
}

def apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_velocity_field(solver, bounds, ax=None, 
                       show_obstacle: bool = True,
                       streamlines: bool = False,
                       title: str = None) -> plt.Axes:
    """
    Plot velocity magnitude field.
    
    Parameters
    ----------
    solver : LBMSolver
        Solver with current flow state
    bounds : TunnelBoundaries
        Geometry definition
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    show_obstacle : bool
        Overlay obstacle outline
    streamlines : bool
        Add streamlines
    title : str, optional
        Custom title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        
    vel_mag = solver.get_velocity_magnitude()
    vel_mag_masked = np.ma.masked_where(bounds.mask, vel_mag)
    
    # Plot velocity magnitude
    im = ax.imshow(vel_mag_masked, origin='lower', cmap='magma',
                   vmin=0, vmax=np.nanmax(vel_mag) * 1.1,
                   aspect='equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Velocity Magnitude $|u|$', 
                       fraction=0.046, pad=0.04)
    
    # Obstacle outline
    if show_obstacle:
        ax.contour(bounds.mask, levels=[0.5], colors='white', 
                  linewidths=1.5, linestyles='-')
    
    # Streamlines
    if streamlines:
        Y, X = np.mgrid[0:bounds.ny, 0:bounds.nx]
        u = solver.u[:, :, 0].copy()
        v = solver.u[:, :, 1].copy()
        u[bounds.mask] = np.nan
        v[bounds.mask] = np.nan
        
        # Sparse streamlines
        ax.streamplot(X, Y, u, v, color='white', density=0.8, 
                     linewidth=0.5, arrowsize=0.5)
    
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    
    if title is None:
        title = f'Velocity Field | Re={solver.reynolds:.0f} | Step {solver.step_count}'
    ax.set_title(title)
    
    return ax


def plot_vorticity_field(solver, bounds, ax=None,
                        vmin: float = None, vmax: float = None,
                        title: str = None) -> plt.Axes:
    """
    Plot vorticity field with diverging colormap.
    
    Parameters
    ----------
    solver : LBMSolver
        Solver with current flow state
    bounds : TunnelBoundaries  
        Geometry definition
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    vmin, vmax : float, optional
        Color scale limits (symmetric around 0)
    title : str, optional
        Custom title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        
    omega = solver.get_vorticity()
    omega_masked = np.ma.masked_where(bounds.mask, omega)
    
    # Symmetric color limits
    if vmax is None:
        vmax = np.nanpercentile(np.abs(omega), 99)
    if vmin is None:
        vmin = -vmax
        
    # Diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im = ax.imshow(omega_masked, origin='lower', cmap='seismic',
                   norm=norm, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, label='Vorticity $\\omega$',
                       fraction=0.046, pad=0.04)
    
    # Obstacle outline
    ax.contour(bounds.mask, levels=[0.5], colors='black',
              linewidths=1.5, linestyles='-')
    
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    
    if title is None:
        title = f'Vorticity Field | Re={solver.reynolds:.0f} | Step {solver.step_count}'
    ax.set_title(title)
    
    return ax


def plot_pressure_field(solver, bounds, ax=None, title: str = None) -> plt.Axes:
    """
    Plot pressure field (from density via equation of state).
    
    Parameters
    ----------
    solver : LBMSolver
        Solver with current flow state  
    bounds : TunnelBoundaries
        Geometry definition
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Custom title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        
    # Pressure from EOS: p = rho * cs^2 = rho / 3
    pressure = solver.rho / 3.0
    p_masked = np.ma.masked_where(bounds.mask, pressure)
    
    # Pressure coefficient: Cp = (p - p_inf) / (0.5 * rho * U^2)
    p_inf = 1.0 / 3.0  # Reference pressure
    q_inf = 0.5 * solver.u_inlet**2
    if q_inf > 1e-10:
        Cp = (pressure - p_inf) / q_inf
        Cp_masked = np.ma.masked_where(bounds.mask, Cp)
        label = 'Pressure Coefficient $C_p$'
        data = Cp_masked
    else:
        label = 'Pressure $p$'
        data = p_masked
    
    # Diverging colormap
    vmax = np.nanpercentile(np.abs(data), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data, origin='lower', cmap='RdBu_r',
                   norm=norm, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, label=label,
                       fraction=0.046, pad=0.04)
    
    ax.contour(bounds.mask, levels=[0.5], colors='black',
              linewidths=1.5, linestyles='-')
    
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    
    if title is None:
        title = f'Pressure Field | Re={solver.reynolds:.0f} | Step {solver.step_count}'
    ax.set_title(title)
    
    return ax


def plot_force_history(force_monitor, ax=None, 
                      title: str = "Aerodynamic Forces") -> plt.Axes:
    """
    Plot time history of aerodynamic forces.
    
    Parameters
    ----------
    force_monitor : ForceMonitor
        Monitor with recorded force history
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str
        Plot title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    steps, drag, lift, cd, cl = force_monitor.get_arrays()
    
    if len(steps) == 0:
        ax.text(0.5, 0.5, 'No data recorded', ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    ax.plot(steps, drag, 'r-', linewidth=1.5, label='Drag', alpha=0.8)
    ax.plot(steps, lift, 'b-', linewidth=1.5, label='Lift', alpha=0.8)
    
    # Zero line for lift reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Force (lattice units)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Annotation for downforce
    mean_forces = force_monitor.get_mean_forces()
    if mean_forces.lift < 0:
        ax.annotate(f'Avg Downforce: {-mean_forces.lift:.4f}',
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    return ax


def plot_energy_spectrum(k: np.ndarray, E_k: np.ndarray, 
                        ax=None, title: str = None,
                        show_theory: bool = True) -> plt.Axes:
    """
    Plot energy spectrum E(k) with theoretical slopes.
    
    Parameters
    ----------
    k : ndarray
        Wave numbers
    E_k : ndarray
        Energy spectrum
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    show_theory : bool
        Show theoretical k^(-5/3) and k^(-3) lines
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Main spectrum
    ax.loglog(k, E_k, 'b.-', linewidth=1.5, markersize=4, 
             label='Simulation', alpha=0.8)
    
    if show_theory and len(k) > 10:
        # Reference point for theoretical lines
        ref_idx = len(k) // 4
        ref_k = k[ref_idx]
        ref_E = E_k[ref_idx]
        
        # Kolmogorov k^(-5/3)
        k_theory = k[k > 0]
        E_kolmogorov = ref_E * (k_theory / ref_k)**(-5/3)
        ax.loglog(k_theory, E_kolmogorov, 'g--', linewidth=1.5, alpha=0.7,
                 label='Kolmogorov $k^{-5/3}$')
        
        # Kraichnan k^(-3) (2D enstrophy cascade)
        E_kraichnan = ref_E * (k_theory / ref_k)**(-3)
        ax.loglog(k_theory, E_kraichnan, 'r:', linewidth=1.5, alpha=0.7,
                 label='Kraichnan $k^{-3}$')
    
    ax.set_xlabel('Wave Number $k$')
    ax.set_ylabel('Energy $E(k)$')
    ax.set_title(title or 'Energy Spectrum')
    ax.legend(loc='best')
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    return ax


def plot_fft_2d(fft_2d: np.ndarray, ax=None, 
               remove_dc: bool = True,
               title: str = "2D Energy Spectrum") -> plt.Axes:
    """
    Plot 2D FFT energy spectrum.
    
    Parameters
    ----------
    fft_2d : ndarray
        Shifted 2D energy spectrum
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    remove_dc : bool
        Remove DC component for better visualization
    title : str
        Plot title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    data = fft_2d.copy()
    
    # Remove DC component (center pixel)
    if remove_dc:
        cy, cx = data.shape[0]//2, data.shape[1]//2
        data[cy, cx] = 0
    
    # Log scale for visibility
    log_data = np.log10(data + 1e-12)
    
    # Dynamic contrast
    vmin = np.percentile(log_data, 50)
    vmax = np.percentile(log_data, 99.9)
    
    im = ax.imshow(log_data, origin='lower', cmap='inferno',
                   vmin=vmin, vmax=vmax, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, label='Log₁₀(Energy)',
                       fraction=0.046, pad=0.04)
    
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_title(title)
    
    return ax


def create_summary_figure(solver, bounds, force_monitor=None,
                         energy_data: Tuple = None,
                         figsize: Tuple[float, float] = (16, 10)) -> plt.Figure:
    """
    Create comprehensive summary figure.
    
    Parameters
    ----------
    solver : LBMSolver
        Solver with current state
    bounds : TunnelBoundaries
        Geometry definition
    force_monitor : ForceMonitor, optional
        Force history monitor
    energy_data : tuple, optional
        (k, E_k, fft_2d) from energy spectrum calculation
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_style()
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2 rows
    # Top row: velocity, vorticity
    # Bottom row: forces, energy spectrum
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)
    
    # Velocity field
    ax1 = fig.add_subplot(gs[0, 0])
    plot_velocity_field(solver, bounds, ax=ax1, streamlines=False)
    
    # Vorticity field
    ax2 = fig.add_subplot(gs[0, 1])
    plot_vorticity_field(solver, bounds, ax=ax2)
    
    # Force history
    ax3 = fig.add_subplot(gs[1, 0])
    if force_monitor is not None and len(force_monitor.history) > 0:
        plot_force_history(force_monitor, ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'Force Monitor\n(no data)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Aerodynamic Forces')
    
    # Energy spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    if energy_data is not None:
        k, E_k, _ = energy_data
        plot_energy_spectrum(k, E_k, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'Energy Spectrum\n(compute with analysis module)',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Energy Spectrum')
    
    # Overall title
    fig.suptitle(f'LBM Simulation Summary | Re={solver.reynolds:.0f} | '
                f'Step {solver.step_count}', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


class AnimationBuilder:
    """
    Build animations from simulation results.
    
    Parameters
    ----------
    solver : LBMSolver
        The solver
    bounds : TunnelBoundaries
        Geometry
    """
    
    def __init__(self, solver, bounds):
        self.solver = solver
        self.bounds = bounds
        self.frames_data = []
        
    def capture_frame(self, force_data: Optional[Tuple] = None):
        """
        Capture current state as animation frame.
        
        Parameters
        ----------
        force_data : tuple, optional
            (drag, lift) values for this frame
        """
        frame = {
            'velocity': self.solver.get_velocity_magnitude().copy(),
            'vorticity': self.solver.get_vorticity().copy(),
            'step': self.solver.step_count,
            'forces': force_data
        }
        self.frames_data.append(frame)
        
    def create_animation(self, field: str = 'velocity',
                        interval: int = 50,
                        figsize: Tuple = (12, 5)) -> FuncAnimation:
        """
        Create animation from captured frames.
        
        Parameters
        ----------
        field : str
            'velocity' or 'vorticity'
        interval : int
            Milliseconds between frames
        figsize : tuple
            Figure size
            
        Returns
        -------
        anim : FuncAnimation
        """
        if not self.frames_data:
            raise ValueError("No frames captured. Call capture_frame() during simulation.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initial frame
        if field == 'velocity':
            data = np.ma.masked_where(self.bounds.mask, self.frames_data[0]['velocity'])
            vmax = max(f['velocity'].max() for f in self.frames_data)
            im = ax.imshow(data, origin='lower', cmap='magma',
                          vmin=0, vmax=vmax, aspect='equal')
            cbar_label = 'Velocity |u|'
        else:  # vorticity
            vmax = max(np.abs(f['vorticity']).max() for f in self.frames_data)
            data = np.ma.masked_where(self.bounds.mask, self.frames_data[0]['vorticity'])
            im = ax.imshow(data, origin='lower', cmap='seismic',
                          vmin=-vmax, vmax=vmax, aspect='equal')
            cbar_label = 'Vorticity ω'
            
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        ax.contour(self.bounds.mask, levels=[0.5], colors='white', linewidths=1)
        title = ax.set_title(f'Step {self.frames_data[0]["step"]}')
        
        def update(frame_idx):
            frame = self.frames_data[frame_idx]
            data = frame[field]
            data_masked = np.ma.masked_where(self.bounds.mask, data)
            im.set_array(data_masked)
            title.set_text(f'Step {frame["step"]}')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=len(self.frames_data),
                            interval=interval, blit=False)
        plt.close(fig)  # Don't display static figure
        
        return anim
    
    def save_animation(self, filename: str, **kwargs):
        """
        Save animation to file.
        
        Parameters
        ----------
        filename : str
            Output filename (e.g., 'sim.gif' or 'sim.mp4')
        **kwargs
            Additional arguments passed to create_animation()
        """
        anim = self.create_animation(**kwargs)
        
        if filename.endswith('.gif'):
            anim.save(filename, writer='pillow', fps=20)
        elif filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=30)
        else:
            anim.save(filename)
            
        print(f"Animation saved to {filename}")
        
    def clear(self):
        """Clear captured frames."""
        self.frames_data.clear()

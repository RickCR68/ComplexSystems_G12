"""
Visualization utilities for LBM simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional, List
import matplotlib.patches as patches


def plot_velocity_field(sim, title: str = "Velocity Field", 
                       figsize: tuple = (12, 4), save_path: Optional[str] = None):
    """
    Plot velocity magnitude field.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    vel_mag = sim.get_velocity_magnitude()
    
    # Mask obstacles
    vel_mag_masked = vel_mag.copy()
    vel_mag_masked[sim.obstacle] = np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vel_mag_masked.T, origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} (Iteration {sim.iteration})')
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, ax


def plot_vorticity(sim, title: str = "Vorticity Field",
                  figsize: tuple = (12, 4), save_path: Optional[str] = None):
    """
    Plot vorticity field.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    vorticity = sim.get_vorticity()
    
    # Mask obstacles
    vorticity_masked = vorticity.copy()
    vorticity_masked[sim.obstacle] = np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Symmetric colormap around zero
    vmax = np.nanmax(np.abs(vorticity_masked))
    im = ax.imshow(vorticity_masked.T, origin='lower', cmap='RdBu_r', 
                   aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} (Iteration {sim.iteration})')
    plt.colorbar(im, ax=ax, label='Vorticity')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, ax


def plot_streamlines(sim, density: int = 2, title: str = "Streamlines",
                    figsize: tuple = (12, 4), save_path: Optional[str] = None):
    """
    Plot streamlines of the flow.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    density : int
        Density of streamlines
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    vel_mag = sim.get_velocity_magnitude()
    
    # Create meshgrid
    x = np.arange(sim.nx)
    y = np.arange(sim.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot velocity magnitude as background
    vel_mag_masked = vel_mag.copy()
    vel_mag_masked[sim.obstacle] = np.nan
    ax.imshow(vel_mag_masked.T, origin='lower', cmap='viridis', alpha=0.6, aspect='auto')
    
    # Plot streamlines
    ax.streamplot(X.T, Y.T, sim.u[0].T, sim.u[1].T, 
                  color='white', density=density, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} (Iteration {sim.iteration})')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, ax


def plot_velocity_profile(y_coords: np.ndarray, u_profile: np.ndarray,
                         u_analytical: Optional[np.ndarray] = None,
                         title: str = "Velocity Profile",
                         figsize: tuple = (6, 4),
                         save_path: Optional[str] = None):
    """
    Plot velocity profile with optional analytical comparison.
    
    Parameters:
    -----------
    y_coords : np.ndarray
        Y coordinates
    u_profile : np.ndarray
        Numerical velocity profile
    u_analytical : np.ndarray, optional
        Analytical velocity profile
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(u_profile, y_coords, 'b-', label='LBM', linewidth=2)
    
    if u_analytical is not None:
        ax.plot(u_analytical, y_coords, 'r--', label='Analytical', linewidth=2)
        ax.legend()
    
    ax.set_xlabel('Velocity u')
    ax.set_ylabel('Y position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, ax


def plot_convergence_history(monitor, figsize: tuple = (12, 4),
                            save_path: Optional[str] = None):
    """
    Plot convergence history from ConvergenceMonitor.
    
    Parameters:
    -----------
    monitor : ConvergenceMonitor
        Convergence monitor object
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot mean velocity
    iterations = np.arange(len(monitor.mean_velocity))
    ax1.plot(iterations, monitor.mean_velocity, 'b-', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Velocity')
    ax1.set_title('Mean Velocity Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot variance
    if monitor.variance_history:
        var_iterations = np.arange(len(monitor.variance_history))
        ax2.semilogy(var_iterations, monitor.variance_history, 'r-', linewidth=1)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Variance (log scale)')
        ax2.set_title('Velocity Variance')
        ax2.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, (ax1, ax2)


def plot_comparison_panel(sim, figsize: tuple = (16, 4),
                         save_path: Optional[str] = None):
    """
    Create a panel showing velocity, vorticity, and streamlines.
    
    Parameters:
    -----------
    sim : LBMSimulation
        Simulation object
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Velocity magnitude
    vel_mag = sim.get_velocity_magnitude()
    vel_mag_masked = vel_mag.copy()
    vel_mag_masked[sim.obstacle] = np.nan
    
    im1 = axes[0].imshow(vel_mag_masked.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title(f'Velocity Magnitude (iter {sim.iteration})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Vorticity
    vorticity = sim.get_vorticity()
    vorticity_masked = vorticity.copy()
    vorticity_masked[sim.obstacle] = np.nan
    vmax = np.nanmax(np.abs(vorticity_masked))
    
    im2 = axes[1].imshow(vorticity_masked.T, origin='lower', cmap='RdBu_r',
                        aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Vorticity')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Streamlines
    x = np.arange(sim.nx)
    y = np.arange(sim.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    axes[2].imshow(vel_mag_masked.T, origin='lower', cmap='viridis', alpha=0.6, aspect='auto')
    axes[2].streamplot(X.T, Y.T, sim.u[0].T, sim.u[1].T,
                      color='white', density=1.5, linewidth=0.5)
    axes[2].set_title('Streamlines')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, axes


def create_animation_from_snapshots(snapshot_files: List[str],
                                   output_file: str = 'animation.mp4',
                                   fps: int = 30):
    """
    Create animation from saved snapshots.
    
    Parameters:
    -----------
    snapshot_files : list
        List of paths to snapshot files
    output_file : str
        Output animation file path
    fps : int
        Frames per second
    """
    # This is a placeholder - implementation would load and animate snapshots
    print(f"Would create animation from {len(snapshot_files)} snapshots")
    print(f"Output: {output_file} at {fps} fps")
    # Full implementation would use matplotlib.animation.FuncAnimation

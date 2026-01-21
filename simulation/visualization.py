"""
Visualization utilities for LBM simulation results.

Provides functions for plotting velocity fields, vorticity, streamlines, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional, List
from matplotlib.figure import Figure


class Visualizer:
    """
    Provides visualization methods for LBM fields.
    """
    
    @staticmethod
    def plot_velocity_magnitude(
        ux: np.ndarray,
        uy: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "Velocity Magnitude",
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "viridis"
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot velocity magnitude field as heatmap.
        
        Args:
            ux: Velocity x-component (ny, nx)
            uy: Velocity y-component (ny, nx)
            mask: Optional obstacle mask (ny, nx) where 1 = solid
            title: Plot title
            figsize: Figure size (width, height)
            cmap: Colormap name
        
        Returns:
            (fig, ax) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        u_mag = np.sqrt(ux**2 + uy**2)
        
        # Mask out obstacles if provided
        if mask is not None:
            u_mag = np.ma.array(u_mag, mask=(mask == 1))
        
        im = ax.imshow(u_mag, cmap=cmap, aspect='auto', origin='lower', interpolation='bilinear')
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Velocity Magnitude')
        
        return fig, ax
    
    @staticmethod
    def plot_vorticity(
        vorticity: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "Vorticity",
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "RdBu_r"
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot vorticity field as heatmap.
        
        Args:
            vorticity: Vorticity field (ny, nx)
            mask: Optional obstacle mask (ny, nx) where 1 = solid
            title: Plot title
            figsize: Figure size
            cmap: Colormap name (diverging colormap recommended)
        
        Returns:
            (fig, ax) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Mask out obstacles if provided
        if mask is not None:
            vorticity = np.ma.array(vorticity, mask=(mask == 1))
        
        # Use symmetric colormap for vorticity (can be positive or negative)
        vmax = np.max(np.abs(vorticity))
        im = ax.imshow(vorticity, cmap=cmap, aspect='auto', origin='lower',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Vorticity')
        
        return fig, ax
    
    @staticmethod
    def plot_velocity_quiver(
        ux: np.ndarray,
        uy: np.ndarray,
        mask: Optional[np.ndarray] = None,
        stride: int = 8,
        title: str = "Velocity Field",
        figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot velocity field as quiver (arrow) plot.
        
        Args:
            ux: Velocity x-component (ny, nx)
            uy: Velocity y-component (ny, nx)
            mask: Optional obstacle mask (ny, nx) where 1 = solid
            stride: Spacing between arrows
            title: Plot title
            figsize: Figure size
        
        Returns:
            (fig, ax) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        y, x = np.mgrid[0:uy.shape[0]:stride, 0:uy.shape[1]:stride]
        u_sub = ux[::stride, ::stride]
        v_sub = uy[::stride, ::stride]
        
        ax.quiver(x, y, u_sub, v_sub, alpha=0.7)
        
        # Overlay obstacle as filled area
        if mask is not None:
            ax.contourf(mask, levels=[0.5, 1.5], colors=['gray'], alpha=0.3)
        
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        return fig, ax
    
    @staticmethod
    def plot_streamlines(
        ux: np.ndarray,
        uy: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "Streamlines",
        figsize: Tuple[int, int] = (12, 6),
        density: float = 2.0
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot streamlines of velocity field.
        
        Args:
            ux: Velocity x-component (ny, nx)
            uy: Velocity y-component (ny, nx)
            mask: Optional obstacle mask (ny, nx) where 1 = solid
            title: Plot title
            figsize: Figure size
            density: Streamline density
        
        Returns:
            (fig, ax) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(ux.shape[1])
        y = np.arange(ux.shape[0])
        xx, yy = np.meshgrid(x, y)
        
        ax.streamplot(xx, yy, ux, uy, density=density, linewidth=0.8, arrowsize=1.5)
        
        # Overlay obstacle
        if mask is not None:
            ax.contourf(xx, yy, mask, levels=[0.5, 1.5], colors=['gray'], alpha=0.3)
        
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('y (lattice units)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        return fig, ax
    
    @staticmethod
    def compute_vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """
        Compute vorticity from velocity field.
        
        ω = ∂uy/∂x - ∂ux/∂y
        
        Uses simple finite differences (second-order central differences at interior).
        
        Args:
            ux: Velocity x-component (ny, nx)
            uy: Velocity y-component (ny, nx)
        
        Returns:
            Vorticity field (ny, nx)
        """
        # Compute gradients
        duy_dx = np.gradient(uy, axis=1)
        dux_dy = np.gradient(ux, axis=0)
        
        vorticity = duy_dx - dux_dy
        
        return vorticity
    
    @staticmethod
    def plot_velocity_profile(
        ux: np.ndarray,
        y_slice: int,
        analytical: Optional[np.ndarray] = None,
        title: str = "Velocity Profile",
        figsize: Tuple[int, int] = (10, 6)
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot 1D velocity profile at a given y position.
        
        Args:
            ux: Velocity x-component (ny, nx)
            y_slice: y-index to extract profile from
            analytical: Optional analytical solution for comparison
            title: Plot title
            figsize: Figure size
        
        Returns:
            (fig, ax) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(ux.shape[1])
        u_profile = ux[y_slice, :]
        
        ax.plot(x, u_profile, 'b-', linewidth=2, label='Numerical')
        
        if analytical is not None:
            ax.plot(x, analytical, 'r--', linewidth=2, label='Analytical')
            ax.legend()
        
        ax.set_xlabel('x (lattice units)')
        ax.set_ylabel('Velocity (lattice units)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax


__all__ = ["Visualizer"]

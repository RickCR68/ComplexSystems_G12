import numpy as np
import matplotlib.pyplot as plt

def compute_energy_spectrum(u, u_field, v_field):
    """
    Calculates the Energy Spectrum E(k) from the velocity field
    using a 2D Fourier Transform.
    Matches Figure 5 in the Dapena-García preprint.
    """
    nx, ny = u_field.shape
    
    # 1. Compute Kinetic Energy Field (Physical Space)
    # KE = 0.5 * (u^2 + v^2)
    # We subtract the mean flow to focus on turbulent fluctuations
    u_fluct = u_field - np.mean(u_field)
    v_fluct = v_field - np.mean(v_field)
    
    # 2. Perform 2D FFT (Frequency Space)
    # This transforms the flow from "Position" to "Wave Number"
    fft_u = np.fft.fft2(u_fluct)
    fft_v = np.fft.fft2(v_fluct)
    
    # Energy Spectral Density = |FFT|^2
    energy_spectrum_2d = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2)
    
    # Shift zero frequency to center for visualization (The "Galaxy" Plot)
    energy_spectrum_shifted = np.fft.fftshift(energy_spectrum_2d)
    
    # 3. Radial Averaging (Collapse 2D -> 1D)
    # We turn the 2D "Galaxy" into the 1D "Slope" plot
    y_freq = np.fft.fftfreq(ny) * ny
    x_freq = np.fft.fftfreq(nx) * nx
    k_grid = np.sqrt(x_freq[np.newaxis, :]**2 + y_freq[:, np.newaxis]**2)
    
    # Flatten and sort by wave number k
    k_flat = k_grid.flatten()
    e_flat = energy_spectrum_2d.flatten()
    
    # Binning (Sum energy in integer k shells)
    k_max = int(min(nx, ny) / 2)
    k_bins = np.arange(1, k_max)
    E_k = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        # Sum all energy roughly at wave number k
        indices = (k_flat >= k - 0.5) & (k_flat < k + 0.5)
        E_k[i] = np.sum(e_flat[indices])
        
    return k_bins, E_k, energy_spectrum_shifted

def plot_complexity_dashboard(solver, step):
    """
    Improved Visuals: Masks the 'Mean Flow' to reveal hidden turbulent structures.
    """
    u_phys = solver.u[:,:,0]
    v_phys = solver.u[:,:,1]
    
    # Get the data
    k, E_k, fft_visual = compute_energy_spectrum(solver.u, u_phys, v_phys)
    
    fig = plt.figure(figsize=(16, 5), dpi=120) # Wider and higher quality
    
    # --- PLOT 1: VORTICITY (The Chaos) ---
    ax1 = fig.add_subplot(131)
    vorticity = np.gradient(v_phys, axis=1) - np.gradient(u_phys, axis=0)
    # Use 'seismic' or 'bwr' centered at 0 to show clockwise vs counter-clockwise spin
    im1 = ax1.imshow(vorticity, cmap='seismic', vmin=-0.08, vmax=0.08, origin='lower')
    ax1.set_title(f"Macroscopic Chaos (Vorticity)\nStep {step}")
    plt.colorbar(im1, ax=ax1, label="Spin Strength", fraction=0.046, pad=0.04)
    
    # --- PLOT 2: FREQUENCY SPACE (The Hidden Structure) ---
    ax2 = fig.add_subplot(132)
    
    # LOGIC FIX: Block the "Sun" (Mean Flow) to see the "Stars" (Turbulence)
    cy, cx = fft_visual.shape[0]//2, fft_visual.shape[1]//2
    fft_visual[cy, cx] = 0  # Remove the massive DC component
    
    # Take Log10 for visibility
    log_fft = np.log10(fft_visual + 1e-12)
    
    # Dynamic Contrast: Focus on the top 95% of waves (ignores background noise)
    vmin = np.percentile(log_fft, 50) 
    vmax = np.percentile(log_fft, 99.9)
    
    im2 = ax2.imshow(log_fft, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title("Frequency Space (The 'Attractor')\n(Mean Flow Removed)")
    plt.colorbar(im2, ax=ax2, label="Log Energy Density", fraction=0.046, pad=0.04)
    
    # --- PLOT 3: ENERGY CASCADE (The Proof) ---
    ax3 = fig.add_subplot(133)
    ax3.loglog(k, E_k, 'b.-', linewidth=2, markersize=8, label='Simulation Data')
    
    # Theoretical Slope k^-3 (Kraichnan 2D Turbulence)
    # Anchor the line to the energy at wave number k=10
    if len(k) > 10:
        ref_idx = 10
        ref_E = E_k[ref_idx]
        ref_k = k[ref_idx]
        # Draw the line
        ax3.loglog(k, ref_E * (k / ref_k)**(-3), 'r--', linewidth=2, label='Theory $k^{-3}$')
        ax3.loglog(k, ref_E * (k / ref_k)**(-5/3), 'g:', linewidth=1.5, alpha=0.7, label='Kolmogorov $k^{-5/3}$')

    ax3.set_xlabel("Wave Number $k$ (Scale)")
    ax3.set_ylabel("Energy $E(k)$")
    ax3.set_title(f"Complexity Proof: Power Law Scaling\nStep {step}")
    ax3.legend()
    ax3.grid(True, which="major", linestyle='-', alpha=0.8)
    ax3.grid(True, which="minor", linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
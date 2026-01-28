import numpy as np

class LBMSolver:
    def __init__(self, nx, ny, reynolds, u_inlet, cs_smag=0.15):
        self.nx = nx
        self.ny = ny
        self.u_inlet = u_inlet
        
        # Physics Parameters
        # Viscosity derived from Re: Re = (U * L) / nu
        self.nu_0 = (u_inlet * (ny/2)) / reynolds
        self.tau_0 = 3.0 * self.nu_0 + 0.5
        self.cs_smag = cs_smag  # Smagorinsky constant
        
        # D2Q9 Constants
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
        self.noslip = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]) # Reverse indices for bounce-back
        
        # Initialization
        self.f = np.zeros((ny, nx, 9))
        self.rho = np.ones((ny, nx))
        self.u = np.zeros((ny, nx, 2))
        self.u[:,:,0] = u_inlet # Initial flow
        
        # Initialize Equilibrium
        self.f = self.equilibrium(self.rho, self.u)

    def equilibrium(self, rho, u):
        """Standard D2Q9 Equilibrium."""
        usqr = u[:,:,0]**2 + u[:,:,1]**2
        feq = np.zeros((self.ny, self.nx, 9))
        for i in range(9):
            cu = u[:,:,0]*self.c[i,0] + u[:,:,1]*self.c[i,1]
            feq[:,:,i] = rho * self.w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)
        return feq

    def collide_and_stream(self, obstacle_mask):
        """
        Main Loop: Collision (with Smagorinsky) -> Streaming -> Bounce-Back
        """
        # 1. Macroscopic properties
        self.rho = np.sum(self.f, axis=2)
        self.u[:,:,0] = np.sum(self.f * self.c[:,0], axis=2) / self.rho
        self.u[:,:,1] = np.sum(self.f * self.c[:,1], axis=2) / self.rho
        
        # 2. Smagorinsky Turbulence Model (Stabilizer)
        # Calculates local stress to increase viscosity in turbulent areas
        # See Eq. 13 in Dapena-Garcia Preprint
        S_xx = np.gradient(self.u[:,:,0], axis=1)
        S_xy = 0.5 * (np.gradient(self.u[:,:,0], axis=0) + np.gradient(self.u[:,:,1], axis=1))
        S_yy = np.gradient(self.u[:,:,1], axis=0)
        Q = np.sqrt(2 * (S_xx**2 + 2*S_xy**2 + S_yy**2))
        
        # Dynamic Tau
        tau_eff = 3.0 * (self.nu_0 + (self.cs_smag**2 * Q)) + 0.5
        tau_eff = np.clip(tau_eff, 0.51, None) # Safety clip
        
        # 3. Collision (BGK)
        feq = self.equilibrium(self.rho, self.u)
        f_post = self.f - (self.f - feq) / tau_eff[:,:,np.newaxis]
        
        # 4. STREAMING
        for i in range(9):
            self.f[:,:,i] = np.roll(np.roll(f_post[:,:,i], self.c[i,0], axis=1), self.c[i,1], axis=0)
            
        # 5. OBSTACLE HANDLING (Robust Bounce-Back)
        # This solves your teammate's "breaking" issue. 
        # We find cells *inside* the obstacle and reverse their velocities.
        boundary_cells = obstacle_mask
        if np.any(boundary_cells):
            bounced_f = self.f[boundary_cells, :][:, self.noslip]
            self.f[boundary_cells, :] = bounced_f
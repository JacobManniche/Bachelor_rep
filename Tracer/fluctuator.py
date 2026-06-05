import numpy as np
import joblib
import os

# Load the POD library once at module level
script_dir = os.path.dirname(__file__)
library_path = os.path.join(script_dir, 'pod_library.joblib')
pod_lib = joblib.load(library_path)


class Fluctuator:
    """
    Fluctuation model supporting multiple methods: 'simple', 'OU', 'Langevin', 'POD'
    
    Can be used with both Euler and RK45 integrators. Compatible with solvers that
    call get_fluctuation_at(seed, pos, time) for evaluation at arbitrary points.
    
    State is maintained internally for time-dependent methods (OU, Langevin).
    """
    
    def __init__(self, method, cf=1.0, C0=2.1, Tg=0.1, n_modes=1, seed=None):
        """
        Initialize the Fluctuator.
        
        Args:
            method: 'simple', 'OU' (Ornstein-Uhlenbeck), 'Langevin', or 'POD'
            cf: Correction factor for fluctuation scaling
            C0: Engineering constant for Langevin model
            Tg: Characteristic time scale for OU model
            n_modes: Number of POD modes to use (only for 'POD' method)
            seed: Random seed for reproducibility
        """
        method = method.lower()
        if method not in ['simple', 'ou', 'langevin', 'pod']:
            raise ValueError(f"Unknown method '{method}'. Must be one of: 'simple', 'OU', 'Langevin', 'POD'")

        self.method = method
        self.cf = cf
        self.C0 = C0
        self.Tg = Tg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if seed == None:
            self.seed = np.random.randint(0, 2**32)  # Base seed for reproducibility
        
        if method in ['ou', 'langevin']:
            self._last_fluctuation = np.zeros(3)  # Initialize last fluctuation
        # Load POD data if needed
        if method == 'pod':
            self.n_modes = n_modes
            self.a = self.rng.normal(loc=0.0, scale=1, size=n_modes)  # Scale random coefficients to match target TKE distribution across modes
            self.phi = pod_lib['Phi']
            self.rans_scale = pod_lib['C']
    
    def get_fluctuation_at(self, pos, tke, epsilon=None, dt=None):
        """
        Get fluctuation at a specific location and time.
        
        This is the main interface for use with Euler and RK45 integrators.
        
        Args:
            pos: Position tuple (x, y, z)
            time: Current time
            tke: Turbulent kinetic energy
            epsilon: Dissipation rate (required for Langevin method)
            dt: Time step (required for OU and Langevin methods)

        Returns:
            Fluctuation vector [u', v', w'] as np.ndarray
        """

        if self.method == 'simple':
            return self._fluctuation_simple(tke)
        
        elif self.method == 'ou':
            fluctuation_old = self._last_fluctuation  # Get previous gust or initialize to zero
            fluctuation_new = self._fluctuation_OU(tke, fluctuation_old, dt)
            self._last_fluctuation = fluctuation_new  # Update state
            return fluctuation_new
        
        elif self.method == 'langevin':
            fluctuation_old = self._last_fluctuation  # Get previous gust or initialize to zero
            fluctuation_new = self._fluctuation_Langevin(tke, epsilon, fluctuation_old, dt)
            self._last_fluctuation = fluctuation_new  # Update state
            return fluctuation_new
        
        elif self.method == 'pod':
            z = pos[2]  # Assuming z is the vertical coordinate
            return self._fluctuation_POD(tke, z)        
        
    def _fluctuation_simple(self, tke):
        """Simple white noise fluctuation model."""
        sigma = np.sqrt(tke * (2.0/3.0))            # std of turbulent kinetic energy with correction factor cf
        fluctuation = self.rng.normal(0.0, sigma, size=3)          # draw three random numbers
        return fluctuation * self.cf

    def _fluctuation_OU(self, tke, fluctuation_old, dt):
        """Ornstein-Uhlenbeck inspired fluctuation model."""
        sigma = np.sqrt(tke * (2.0/3.0))                                            # std of turbulent kinetic energy
        eta = self.rng.normal(0.0, 1.0, size=3)                                          # draw three random numbers
        fluctuation = fluctuation_old*(1.0 - dt/self.Tg) + np.sqrt(2.0*sigma**2*dt/self.Tg) * eta    # new gust = decaying previous gust term + new random component
        return fluctuation * self.cf

    def _fluctuation_Langevin(self, tke, epsilon, fluctuation_old, dt):
        """Langevin implementation of fluctuation model."""
        gamma = (3.0/4.0) * self.C0 * (epsilon/tke) if tke > 0 else 0                   # decay term based on engineering factor C0, tke, and epsilon
        eta = self.rng.normal(0.0, 1.0, size=3)                                          # random number draw 
        fluctuation = fluctuation_old * (1.0 - gamma * dt) + np.sqrt(self.C0 * epsilon * dt) * eta     
        
        return fluctuation * self.cf

    def _fluctuation_POD(self, tke, z):
        """POD-based fluctuation model."""
        sigma_target = np.sqrt(tke * self.rans_scale[:self.n_modes])  # Target std for each mode based on TKE and mode energy fraction
        a_local = self.a * sigma_target  # Scale random coefficients to match target TKE distribution across modes
        if self.n_modes == 1:
            fluctuation = a_local[0] * self.phi(z)[0]  # Use only the first mode
        else:
            fluctuation = a_local @ self.phi(z)[:self.n_modes]  # Synthesize fluctuation using one POD mode and coefficient

        return fluctuation * self.cf
    
    def plot(self, range=(0, 100), num_points=100, tke=1.0, epsilon=1.0, ax=None):
        """
        Utility function to visualize the fluctuation profile for constant TKE.
        if method is POD, it will show the fluctuation profile across the specified z range.
        if method is simple, OU, or Langevin, it will show the fluctuation profile across the specified time range.
        """
        import matplotlib.pyplot as plt
        
        # Create figure and axes if not provided
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        values = np.linspace(range[0], range[1], num_points)
        if self.method == 'pod':
            fluctuations = np.array([self._fluctuation_POD(tke, z) for z in values])  # Use tke=1.0 for visualization
            ax.plot(fluctuations[:, 0], values, label="u' (x-fluctuation)")
            ax.plot(fluctuations[:, 1], values, label="v' (y-fluctuation)")
            ax.plot(fluctuations[:, 2], values, label="w' (z-fluctuation)")
            ax.set_ylabel('Height (z)')
            ax.set_xlabel('Fluctuation')
        elif self.method in ['simple', 'ou', 'langevin']:
            fluctuations = np.array([self.get_fluctuation_at((0, 0, 0), tke=tke, epsilon=epsilon, dt=values[1]) for t in values])  # Use tke=1.0 for visualization
            ax.plot(values, fluctuations[:, 0], label="u' (x-fluctuation)")
            ax.plot(values, fluctuations[:, 1], label="v' (y-fluctuation)")
            ax.plot(values, fluctuations[:, 2], label="w' (z-fluctuation)")
            ax.set_xlabel('Time')
            ax.set_ylabel('Fluctuation')
        ax.legend()
        ax.grid()

        if not ax:
            plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    fluc = Fluctuator(method='pod')
    fluc.plot(range=(0, 100), num_points=200, tke=10)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    fluc_ou = Fluctuator(method='OU', dt=0.01, Tg=0.1)
    fluc_ou.plot(range=(0, 100), num_points=200, tke=10, ax=ax[0])

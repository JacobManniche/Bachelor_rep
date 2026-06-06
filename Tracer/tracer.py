import numpy as np
from Tracer.windfield import WindField
from Tracer.solvers import solver_rk45, solver_euler

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([np.cos(theta), 0, np.sin(theta)])
    return V0

def initial_spin_rate(spin_rate, spin_axis=0):
    """Returns the initial spin rate in radians per second given the spin rate in rpm"""
    phi = np.radians(spin_axis)
    # Initial Spin (W0) in rad/s
    w_mag = spin_rate * (2 * np.pi / 60)
    
    # For X-axis flight; X: rifle, Y: Backspin, Z: Sidespin 
    w0 = w_mag * np.array([0,-np.cos(phi), np.sin(phi)])

    return w0
class Trajectory:
    def __init__(self, ball_speed, launch_angle, spin_rate, spin_axis=0, orientation=0, P0=np.array([0, 0, 0]), wind=None, fluc=None):
        """
        Initializes the Trajectory object with initial conditions and wind field.
        Parameters:
        - ball_speed: Initial speed of the ball in m/s
        - launch_angle: Launch angle of the ball in degrees
        - spin_rate: Spin rate of the ball in rpm
        - spin_axis: Spin axis angle in degrees (default: 0 for pure backspin)
        - P0: Initial position vector (m)
        - orientation: Orientation angle in degrees (default: 0)
        - wind: WindField object that provides the get_velocity_at(x, y, z)
        - fluc: Fluctuator object that provides the get_fluctuation_at(pos, tke, epsilon) method
        """
        self.args = (ball_speed, launch_angle, spin_rate, spin_axis, orientation)
        self.P0 = P0
        self.V0 = initial_velocity(speed=ball_speed, angle=launch_angle)
        self.W0 = initial_spin_rate(spin_rate=spin_rate, spin_axis=spin_axis)
        if orientation != 0:
            self.rotate(orientation)
        if wind is None:
            print("No wind provided. Using default uniform wind with U_ref=0.")
            wind = WindField(profile='uniform', U_ref=0)
        self.wind = wind
        self.fluc = fluc

        self.is_solved = False

    def rotate(self, angle):
        """Rotates the initial velocity and spin vectors by a given angle in degrees around the Z-axis."""
        psi = np.radians(angle)
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0,            0,           1]])
        self.V0 = Rz @ self.V0
        self.W0 = Rz @ self.W0

        self.is_solved = False
        
        return self

    def solve(self, solver='rk45', dt=0.01, **kwargs):
        """
        Solves the trajectory of the ball using the specified solver method.
        Parameters:
        - solver: 'rk45' for Runge-Kutta 4(5) method, 'euler' for simple Euler method.
        - dt: Time step for the solver (only visual for rk45 can be set to None).
        - kwargs: Additional parameters to pass to the solver 
            decay_rate: Decay rate for spin (default: 0.05)
            rtol: Relative tolerance for RK45 (default: 1e-6)
            mt: Max time for rk45 solver (default: 15 seconds)
        Returns:
        - t: Array of time steps
        - p: Array of positions at each time step
        - v: Array of velocities at each time step
        - w: Array of spin rates at each time step
        """
        if solver == 'rk45':
            if self.fluc and self.fluc.method in ['ou', 'langevin', 'simple']:
                raise ValueError("RK45 solver is not compatible with temporal stochastic turbulence methods (OU, Langevin, Simple). Please use 'euler' solver for these methods.")
            t, p, v, w = solver_rk45(self.V0, self.W0, P0=self.P0, wind=self.wind, fluc=self.fluc, dt=dt, **kwargs)
        elif solver == 'euler':
            t, p, v, w = solver_euler(self.V0, self.W0, P0=self.P0, wind=self.wind, fluc=self.fluc, dt=dt, **kwargs)
        else:
            raise ValueError("Invalid solver specified. Use 'rk45' or 'euler'.")
        
        self.t = t
        self.p = p
        self.v = v
        self.w = w
        self.traj = (t, p, v, w)

        self.is_solved = True

        return t, p, v, w

    def plot(self):
        from Tracer.debug_tools import plot_trajectories
        plot_trajectories([self])

    def animate(self):
        from Tracer.animate import animate
        animate(self)

    def __repr__(self):
        if self.is_solved:
            return f"Trajectory(ball_speed={self.args[0]}, launch_angle={self.args[1]}, spin_rate={self.args[2]}, spin_axis={self.args[3]}) \nV0={self.V0.round(2)}, \nW0={self.W0.round(2)}, \nP0={self.P0.round(2)}, \nFinal Position={self.p[-1].round(2)}, \n Time={self.t[-1]:.2f} s"
        else:
            return f"Trajectory(V0={self.V0.round(2)}, W0={self.W0.round(2)}, P0={self.P0.round(2)})"

from typing import Literal

# Define the allowed clubs for IDE autocomplete suggestions
# (Note: "3 Iron" removed as it is only present in PGA, not LPGA)
Clubs = Literal["Driver", "3-wood", "5-wood", "Hybrid", "4 Iron", "5 Iron", "6 Iron", "7 Iron", "8 Iron", "9 Iron", "PW"]

# Dict format narrowed down to strictly: [Ball Speed (mph), Launch Angle (deg), Spin Rate (rpm)]
pga_data = {'Driver': [76.44384, 10.4, 2545],
 '3-wood': [72.42048, 9.3, 3663],
 '5-wood': [69.73824, 9.7, 4322],
 'Hybrid': [66.60896, 10.2, 4587],
 '3 Iron': [64.8208, 10.3, 4404], # Note: 3 Iron is only present in PGA data, not in LPGA data
 '4 Iron': [62.5856, 10.8, 4782],
 '5 Iron': [60.3504, 11.9, 5280],
 '6 Iron': [58.1152, 14.0, 6204],
 '7 Iron': [54.98592, 16.1, 7124],
 '8 Iron': [52.75072, 17.8, 8078],
 '9 Iron': [50.06848, 20.0, 8793],
 'PW': [46.49216, 23.7, 9316]}


lpga_data = {'Driver': [63.926719999999996, 12.6, 2506],
 '3-wood': [60.3504, 11.6, 2595],
 '5-wood': [58.1152, 12.3, 4320],
 'Hybrid': [55.88, 13.9, 4504],
 '4 Iron': [52.75072, 13.9, 4608],
 '5 Iron': [50.962559999999996, 14.6, 4966],
 '6 Iron': [49.62144, 16.7, 5904],
 '7 Iron': [47.38624, 18.5, 6630],
 '8 Iron': [45.598079999999996, 20.8, 7413],
 '9 Iron': [42.4688, 23.5, 7605],
 'PW': [39.33952, 25.2, 8465]}


class PGA(Trajectory):
    def __init__(
        self, 
        club: Clubs, 
        l: bool = False, 
        spin_axis: float = 0, 
        orientation: float = 0, 
        P0: np.ndarray = np.array([0, 0, 0]), 
        wind=None, 
        fluc=None
    ):
        self.club = club
        self.data = pga_data[club] if not l else lpga_data[club]
        
        super().__init__(
            ball_speed=self.data[0],
            launch_angle=self.data[1],
            spin_rate=self.data[2],
            spin_axis=spin_axis,
            orientation=orientation,
            P0=P0,
            wind=wind,
            fluc=fluc
        )
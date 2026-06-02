from Tracer.windfield import WindField
from Tracer.fluctuator import Fluctuator
import numpy as np
from scipy.integrate import solve_ivp

def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

# Constants for acceleration calculations
r=0.0214
m=0.046
rho=1.204
g=9.81
mu = 1.82e-5 #Tabel B.3 for mu and rho
Ab = np.pi * r**2 # cross-sectional area of the ball
constant_property = Ab * rho/(2 * m) # constant property for efficiency
G = -g * np.array([0, 0, 1]) # gravity vector

def acc(V, W, wind):

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)

    cd, cl = coefficients(U_mag, norm(W)) # get drag and lift coefficients based on current conditions
    
    # Aerodynamic forces

    # The "Standard Aerodynamic" Model
    D = -constant_property * cd * U_mag * U
    
    # The "Standard Aerodynamic" Model
    L = constant_property * cl * U_mag**2 * UW_cross / UW_cross_norm if UW_cross_norm > 0 else np.array([0, 0, 0])
    
    a = D + L + G # acceleration from Newton's second law

    return a


# Constants from the Slazenger ball study
CD1 = 0.24
CD2 = 0.18
CD3 = 0.06
CL1 = 0.54
A1 = 90000.0
A2 = 200000.0

def coefficients(v, w):
    """
    Aerodynamic model restricted to driver shots (Re 70k-210k).
    v: velocity (m/s)
    w: spin rate (rad/s)
    """
    
    # Calculate Reynolds Number and Spin Parameter
    re = rho * (v * (2 * r)) / mu
    s = (w * r) / v if v > 0 else 0  # spin parameter, avoid division by zero
    
    # Lift Coefficient
    cl = CL1 * (max(s, 0)**0.4)
    
    # Drag Coefficient:
    drag_oscillation = CD3 * np.sin(np.pi * (re - A1) / A2)
    cd = CD1 + (CD2 * s) + drag_oscillation
    
    return cd, cl



def odesystem(t, y, W_initial, wind: WindField, fluc: Fluctuator, decay_rate):
    # Unpack the state
    p = y[:3]
    v = y[3:]
    
    # Get current spin rate (Decay is time-dependent)
    w = W_initial * np.exp(-decay_rate * t)
    
    # Get Acceleration
    wind_vec, tke, eps = wind.get_profile_at(*p)  # Get local wind velocity vector

    if np.isnan(wind_vec).any():
        print(f"Warning: NaN in wind at position {p.round(3)}. Setting wind to zero.")
        wind_vec = np.zeros(3)  # Fallback to zero wind if interpolation fails

    if fluc:
        fluctuation = fluc.get_fluctuation_at(pos=p, tke=tke, epsilon=eps)  # Get turbulence fluctuation vector
        wind_vec += fluctuation  # Add turbulence to the wind vector

    a = acc(v, w, wind_vec)
    
    return np.concatenate([v, a])

# Define the 'Ground Hit' event to stop the solver early

def hit_ground(t, y, *args):
    return y[2] # The Z-coordinate
hit_ground.terminal = True
hit_ground.direction = -1

def solver_rk45(V0, W0, P0, wind: WindField, fluc: Fluctuator=None, dt=0.05, decay_rate=0.05, mt=15, rtol=1e-6):
    """
    Solves the equations of motion for the ball given the initial conditions and parameters using the RK45 method.
    It will return the trajectory of the ball as a function of time.
    Parameters:
    - V0: Initial velocity vector (m/s)
    - W0: Initial spin vector (rad/s)
    - P0: Initial position vector (m)
    - wind: WindField object that provides the get_velocity_at(x, y, z) method
    - fluc: Fluctuator object that provides the get_fluctuation_at(pos, tke, epsilon) method
    - dt: Time step for the solver (used for t_eval, can be set to None for adaptive time steps)
    - decay_rate: Decay rate for spin (default: 0.05)
    - mt: Max time for the solver to run (default: 15 seconds)
    - rtol: Relative tolerance for the RK45 solver (default: 1e-6
    Returns:
    - t_steps: Array of time steps at which the solution was evaluated
    - positions: Array of shape (N, 3) containing the position of the ball
    - velocities: Array of shape (N, 3) containing the velocity of the ball
    - spin_rates: Array of shape (N, 3) containing the spin rate of
    """
    # Similar setup as above but with fixed time steps
    # Initial state: Combine P0 and V0
    y0 = np.concatenate([P0, V0])

    t_requested = np.arange(0, mt + dt, dt) if dt else None

    sol = solve_ivp(
        odesystem, 
        t_span=(0, mt),       # Max time 15 seconds
        y0=y0, 
        method='RK45',        # Dormand-Prince adaptive solver
        args=(W0, wind, fluc, decay_rate),
        events=hit_ground,    # Stops the moment Z hits 0
        t_eval=t_requested,
        rtol=rtol, atol=rtol/1000  # Precision settings
    )

    # Accessing results:
    t_steps = sol.t
    positions = sol.y[:3, :].T # Shape (N, 3)
    velocities = sol.y[3:, :].T
    spin_rates = W0 * np.exp(-decay_rate * t_steps[:, np.newaxis])

    return t_steps, positions, velocities, spin_rates

def solver_euler(V0, W0, P0, wind: WindField, fluc: Fluctuator=None, dt=0.01, decay_rate=0.05):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = [0]
    P = [P0] # position array
    V = [V0]
    W = [W0] # initial angular velocity (rad/s)

    while P[-1][2] >= 0: # while the ball is above the ground
        
        p = P[-1]
        wind_vec, tke, eps = wind.get_profile_at(*p)  # Get local wind velocity vector
        
        if np.isnan(wind_vec).any():
            print(f"Warning: NaN in wind at position {p.round(3)}. Setting wind to zero.")
            wind_vec = np.zeros(3)  # Fallback to zero wind if interpolation fails
        
        if fluc:
            fluctuation = fluc.get_fluctuation_at(pos=p, tke=tke, epsilon=eps, dt=dt)  # Get turbulence fluctuation vector
            wind_vec += fluctuation  # Add turbulence to the wind vector

        a = acc(V[-1], W[-1], wind_vec)

        V.append(V[-1] + a * dt) # velocity
        P.append(P[-1] + V[-1] * dt) # position
        W.append(W[-1] * np.exp(-decay_rate * dt)) # spin rate
        t.append(t[-1] + dt) # time

    return np.array(t), np.array(P), np.array(V), np.array(W)


if __name__ == "__main__":
    from Tracer import initial_spin_rate, initial_velocity
    from Tracer.debug_tools import plot_trajectories
    # Example usage
    P0 = np.array([-20, 0, 0])  # Initial position
    V0 = initial_velocity(speed=76.44384, angle=10.4)  # Initial velocity vector
    W0 = initial_spin_rate(spin_rate=2545, spin_axis=1.25)  # Initial spin vector
    wind_synt = WindField(profile='log', direction=0, U_ref=8, z_ref=90, z0=0.03)

    file = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/RANS/nc files/flow_flat_2m_2m.nc'
    wind_rans = WindField(ds=file, profile='rans', scale_factor=8)

    plot_trajectories([solver_rk45(V0,W0,P0=P0,wind=wind_rans), solver_euler(V0,W0,P0=P0,wind=wind_rans)])
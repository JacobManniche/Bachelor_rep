from Tracer.solvers import norm, coefficients
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(trajectories, plot=True):
    ax = plt.subplots(3, 1, figsize=(10, 8))[1]
    for traj in trajectories:
        traj = Trajectory(76, 12, 2000, 1.2)
        traj.solve(solver='euler', dt=0.01)
        ax[0].plot(traj.p[:, 0], traj.p[:, 2]) # Plot height vs time
        ax[1].plot(traj.p[:, 0], traj.p[:, 1]) # Plot horizontal distance vs time
        ax[2].plot(traj.t, traj.w[:, 1])

    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Height (m)')
    ax[0].set_title('Trajectory of the Golf Ball')
    ax[0].axis('equal')
    ax[0].grid()
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Horizontal Distance (m)')
    ax[1].set_title('Horizontal Trajectory of the Golf Ball')
    ax[1].axis('equal')
    ax[1].grid()
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Spin Rate (rad/s)')
    ax[2].set_title('Spin Rate Decay Over Time')
    ax[2].grid()

    if plot:
        plt.tight_layout()
        plt.show()
    else:
        return ax

def plot_coefficients(trajectories, plot=True):
    mu = 1.82e-5 # Dynamic viscosity of air (kg/(m·s))
    r = 0.0214 # Radius of the golf ball (m)
    
    ax = plt.subplots(2, 2, figsize=(10, 8))[1]
    for traj in trajectories:
        traj = Trajectory(76, 12, 2000, 1.2)
        traj.solve(solver='euler', dt=0.01)
        re = [(norm(traj.v[i]) * (2*r)) / mu for i in range(len(traj.v))] # Reynolds Number calculation
        s = [norm(traj.w[i]) * r / norm(traj.v[i]) if norm(traj.v[i]) > 0 else 0 for i in range(len(traj.v))] # Spin parameter
        cd = [coefficients(norm(traj.v[i]), norm(traj.w[i]))[0] for i in range(len(re))]
        cl = [coefficients(norm(traj.v[i]), norm(traj.w[i]))[1] for i in range(len(re))]
        
        ax[0, 0].plot(re, cd) # Plot height vs time
        ax[1, 0].plot(s, cl) # Plot horizontal distance vs time

        ax[0, 1].plot(t, cd) # Plot height vs time
        ax[1, 1].plot(t, cl) # Plot horizontal distance vs time
    
    ax[0, 0].set_xlabel('Reynolds Number')
    ax[0, 0].set_ylabel('Drag Coefficient (Cd)')
    ax[0, 0].set_title('Drag Coefficient vs Reynolds Number')
    ax[0, 0].grid()
    ax[1, 0].set_xlabel('Spin Parameter (s)')
    ax[1, 0].set_ylabel('Lift Coefficient (Cl)')
    ax[1, 0].set_title('Lift Coefficient vs Spin Parameter')
    ax[1, 0].grid()
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('Drag Coefficient (Cd)')
    ax[0, 1].set_title('Drag Coefficient Decay Over Time')
    ax[0, 1].grid()
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Lift Coefficient (Cl)')
    ax[1, 1].set_title('Lift Coefficient Decay Over Time')
    ax[1, 1].grid()
    
    if plot:
        plt.tight_layout()
        plt.show()
    else:
        print("Returning axes for further manipulation...")
        return ax

if __name__ == "__main__":
    from Tracer import Trajectory
    t = []
    for decay in [0.05, 0.1, 0.2, 1]:
        t.append(Trajectory(76, 12, 2000, 1.2).solve())
    plot_trajectories(t)
    plot_coefficients(t)


    df_pga_data = np.array([[7.644384e+01, 1.040000e+01, 2.545000e+03, 3.200000e+01,
        2.580000e+02],
       [7.242048e+01, 9.300000e+00, 3.663000e+03, 2.900000e+01,
        2.280000e+02],
       [6.973824e+01, 9.700000e+00, 4.322000e+03, 3.000000e+01,
        2.160000e+02],
       [6.660896e+01, 1.020000e+01, 4.587000e+03, 2.800000e+01,
        2.110000e+02],
       [6.482080e+01, 1.030000e+01, 4.404000e+03, 2.700000e+01,
        1.990000e+02],
       [6.258560e+01, 1.080000e+01, 4.782000e+03, 2.800000e+01,
        1.920000e+02],
       [6.035040e+01, 1.190000e+01, 5.280000e+03, 3.000000e+01,
        1.820000e+02],
       [5.811520e+01, 1.400000e+01, 6.204000e+03, 2.900000e+01,
        1.720000e+02],
       [5.498592e+01, 1.610000e+01, 7.124000e+03, 3.100000e+01,
        1.610000e+02],
       [5.275072e+01, 1.780000e+01, 8.078000e+03, 3.000000e+01,
        1.500000e+02],
       [5.006848e+01, 2.000000e+01, 8.793000e+03, 2.900000e+01,
        1.390000e+02],
       [4.649216e+01, 2.370000e+01, 9.316000e+03, 2.900000e+01,
        1.300000e+02]])
    
    # t = []
    # for bs, la, sr, mh, c in df_pga_data[:4]:
    #     V0 = initial_velocity(speed=bs, angle=la)
    #     W0 = np.array([200, -sr, -10])
    #     t.append(solver(P0, V0, W0, wind, dt=0.01))
    # ax = plot_trajectory(t, plot=False)
    # plt.show()
    # plot_coefficients(t)
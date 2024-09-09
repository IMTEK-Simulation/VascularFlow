import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from VascularFlow.two_step_Lax_Wendroff_method import lax_wendroff
from VascularFlow.eigenvalues_computation import compute_lambda
from VascularFlow.global_characteristic_variables import compute_W


def test_single_channel_unsteady_flow_inlet_flow_rate(plot=True):
    """Test cross-sectional area and flow rate change with time across a single compliant channel."""
    tube_length = 0.1  # float, length of the single channel (m)
    nb_nodes = 101  # int, number of spatial points
    dx = tube_length / (nb_nodes - 1)  # float, spatial step size
    x = np.linspace(0, tube_length, nb_nodes)  # np.Array, x axis coordinate
    alpha = 1  # float, axial momentum flux correction coefficient (or Coriolis coefficient)
    channel_elasticity = 265.86  # float, channel elasticity (in the second p-A model)
    density = 1000  # float, fluid density
    kinematic_viscosity = 3.5e-6  # float, fluid viscosity
    # Boundary and initial conditions
    Q0 = 0.0002144 / 2  # flow rate at the inlet of the channel
    A0 = 7.85e-5 / 2  # initial cross-sectional area of the channel
    # Initialize arrays for flow rate and cross-sectional area
    Q = np.zeros(nb_nodes)
    A = np.full(nb_nodes, A0)
    # Time stepping
    dt = 1e-4  # float, temporal step size
    nt = 200  # int, number of temporal points
    # Results storage for animation
    A_results = []
    Q_results = []
    # compute incoming characteristic variable (W2) at the outlet of the channel
    W2_outlet = Q[-1] / A[-1] - 4 * np.sqrt(channel_elasticity / (2 * density * A0)) * (A[-1] ** 0.25)
    for n in range(nt):
        A_old = A.copy()
        Q_old = Q.copy()
        # compute the eigenvalues and characteristic variables
        lambda1 = compute_lambda(A_old, Q_old, A0, density, channel_elasticity)[0]
        lambda2 = compute_lambda(A_old, Q_old, A0, density, channel_elasticity)[1]
        W1 = compute_W(A_old, Q_old, A0, density, channel_elasticity)[0]
        W2 = compute_W(A_old, Q_old, A0, density, channel_elasticity)[1]
        A = lax_wendroff(A_old, Q_old, A0, nb_nodes, dx, dt, alpha, channel_elasticity, density, kinematic_viscosity)[0]
        Q = lax_wendroff(A_old, Q_old, A0, nb_nodes, dx, dt, alpha, channel_elasticity, density, kinematic_viscosity)[1]
        # compute boundary values at the both ends of the computational domain using the global characteristic variables
        # compute outgoing characteristic variable (W2) at the inlet of the channel
        interpolation_function_1 = interp1d(x, W2, kind='linear', fill_value="extrapolate")
        W2_inlet = interpolation_function_1(-dt * lambda2[0])
        # compute outgoing characteristic variable (W1) at the outlet of the channel
        interpolation_function_2 = interp1d(x, W1, kind='linear', fill_value="extrapolate")
        W1_outlet = interpolation_function_2(tube_length - dt * lambda1[-1])
        # compute incoming characteristic variable (W1) at the inlet of the channel
        W1_inlet = -W2_inlet + 2 * (Q0 / A_old[0])

        # updating the boundary values at the both ends of the computational domain
        Q[0] = Q0
        A[0] = ((2 * density * A0) / channel_elasticity) ** 2 * ((W1_inlet - W2_inlet) / 8) ** 4
        A[-1] = ((2 * density * A0) / channel_elasticity) ** 2 * ((W1_outlet - W2_outlet) / 8) ** 4
        Q[-1] = (A[-1] / 2) * (W1_outlet + W2_outlet)

        A_results.append(A.copy())
        Q_results.append(Q.copy())
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.set_xlim([0, tube_length])
        ax1.set_ylim([0, 0.0002])
        ax2.set_xlim([0, tube_length])
        ax2.set_ylim([0, 0.0005])
        line1, = ax1.plot(x, A_results[0], lw=2, label='Area (A)')
        line2, = ax2.plot(x, Q_results[0], lw=2, label='Flow Rate (Q)')
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

        def animate(i):
            line1.set_ydata(A_results[i])
            line2.set_ydata(Q_results[i])
            time_text.set_text(f'Time = {i * dt:.2f} s')
            return line1, line2, time_text

    ani = FuncAnimation(fig, animate, frames=nt, interval=5, blit=True)
    ani.save("single_channel_unsteady_flow_inlet_flow_rate.gif")
    plt.xlabel('x')
    ax1.set_ylabel('Area (A)')
    ax2.set_ylabel('Flow Rate (Q)')
    ax1.set_title('Variation of A and Q with respect to x')
    ax2.set_title('Variation of A and Q with respect to x')
    plt.tight_layout()
    plt.show()

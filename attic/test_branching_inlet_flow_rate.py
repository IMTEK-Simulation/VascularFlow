import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from scipy.optimize import root

from VascularFlow.Numerics.two_step_Lax_Wendroff_method import lax_wendroff
from VascularFlow.Numerics.eigenvalues_computation import compute_lambda
from VascularFlow.global_characteristic_variables import compute_W

# in this code we want to find the unknowns cross-sectional area (A) and flow rate (Q) in a
# One-dimensional model of bifurcation by domain decomposition technique. The governing equations are:
# 1) The mass conservation equation:     ∂A/∂t + ∂Q/∂x = 0
# 2) The momentum conservation equation: ∂Q/∂t + ∂/∂x(αQ^2/A) + A/ρ ∂p/∂x + kappa Q/A = 0
# 3) The pressure-area relationship:     P-P0 = beta/A0 (√A - √A0); beta= Eh0√π
# Note: if we plug Eq.3 into Eq.2 >>>    ∂Q/∂t + ∂/∂x(αQ^2/A) + (ß/2 ρ A_0) √A ∂A/∂x + kappa Q/A = 0


def test_branching_inlet_flow_rate(plot=True):
    """Test cross-sectional area and flow rate change with time in one-dimensional model of bifurcation
     by domain decomposition technique."""

    # length of the three channels _ parent and daughter vessels
    tube_length_P = 0.1    # length of the parent vessel
    tube_length_D1 = 0.1   # length of the daughter1 vessel
    tube_length_D2 = 0.1   # length of the daughter2 vessel
    nb_nodes_P = 101   # parent vessel number of nodes
    nb_nodes_D1 = 101  # daughter1 vessel number of nodes
    nb_nodes_D2 = 101  # daughter2 vessel number of nodes

    # spatial grid in parent and daughter vessels
    def dx(nx, L):
        return L / (nx - 1)
    # coordinate in x direction in parent and daughter vessels

    def x(nx, L):
        return np.linspace(0, L, nx)

    # Boundary conditions
    Q0 = 0.0002144  # flow rate at the inlet of parent vessel
    # Initial conditions
    A0_P = 7.85e-5       # initial cross-sectional area in parent vessel
    A0_D1 = 7.85e-5 / 2  # initial cross-sectional area in daughter1 vessel
    A0_D2 = 7.85e-5 / 2  # initial cross-sectional area in daughter2 vessel
    # initial cross-sectional area array in vessels
    A_P = np.full(nb_nodes_P, A0_P)
    A_D1 = np.full(nb_nodes_D1, A0_D1)
    A_D2 = np.full(nb_nodes_D2, A0_D2)
    # initial flow rate array in vessels
    Q_P = np.zeros(nb_nodes_P)
    Q_D1 = np.zeros(nb_nodes_D1)
    Q_D2 = np.zeros(nb_nodes_D2)
    alpha = 1  # the momentum correction coefficient in momentum equation
    # physical and mechanical characteristics of the vessels in pressure-area relationship
    # P-P0 = beta/A0 (√A - √A0); beta= Eh0√π
    channel_elasticity_P = 265.86  # elasticity of parent vessel
    channel_elasticity_D1 = 265.86  # elasticity of daughter1 vessel
    channel_elasticity_D2 = 265.86  # elasticity of daughter2 vessel
    # properties of fluid
    density = 1000  # fluid density
    kinematic_viscosity = 3.5e-6  # fluid kinematic viscosity

    # Time stepping
    dt = 1e-4
    nt = 1000

    # Results storage for animation
    A_P_results = []
    A_D1_results = []
    A_D2_results = []

    # non-reflecting conditions at the outlet of daughter vessels
    # the incoming characteristics of the daughter vessels at the outlet of daughter vessels
    W2_D1_outlet = Q_D1[-1] / A_D1[-1] - 4 * np.sqrt(channel_elasticity_D1 / (2 * density * A0_D1)) * (A_D1[-1] ** 0.25)
    W2_D2_outlet = Q_D2[-1] / A_D2[-1] - 4 * np.sqrt(channel_elasticity_D2 / (2 * density * A0_D2)) * (A_D2[-1] ** 0.25)
    for n in range(nt):
        A_old_P = A_P.copy()
        A_old_D1 = A_D1.copy()
        A_old_D2 = A_D2.copy()

        Q_old_P = Q_P.copy()
        Q_old_D1 = Q_D1.copy()
        Q_old_D2 = Q_D2.copy()
        ###############################################################################################################
        # updating the values of cross-sectional area and flow rate at the internal nodes of the parent vessel
        # by two-step Lax-Wendroff method
        A_P = lax_wendroff(
             A_old_P, Q_old_P, A0_P, nb_nodes_P, dx(nb_nodes_P, tube_length_P), dt, alpha, channel_elasticity_P,
             density, kinematic_viscosity)[0]
        Q_P = lax_wendroff(
             A_old_P, Q_old_P, A0_P, nb_nodes_P, dx(nb_nodes_P, tube_length_P), dt, alpha, channel_elasticity_P,
             density, kinematic_viscosity)[1]
        # updating the values of cross-sectional area and flow rate at the internal nodes of the first daughter vessel
        # by two-step Lax-Wendroff method
        A_D1 = lax_wendroff(
            A_old_D1, Q_old_D1, A0_D1, nb_nodes_D1, dx(nb_nodes_D1, tube_length_D1), dt, alpha, channel_elasticity_D1,
            density, kinematic_viscosity)[0]
        Q_D1 = lax_wendroff(
            A_old_D1, Q_old_D1, A0_D1, nb_nodes_D1, dx(nb_nodes_D1, tube_length_D1), dt, alpha, channel_elasticity_D1,
            density, kinematic_viscosity)[1]
        # updating the values of cross-sectional area and flow rate at the internal nodes of the second daughter vessel
        # by two-step Lax-Wendroff method
        A_D2 = lax_wendroff(
            A_old_D2, Q_old_D2, A0_D2, nb_nodes_D2, dx(nb_nodes_D2, tube_length_D2), dt, alpha, channel_elasticity_D2,
            density, kinematic_viscosity)[0]
        Q_D2 = lax_wendroff(
            A_old_D2, Q_old_D2, A0_D2, nb_nodes_D2, dx(nb_nodes_D2, tube_length_D2), dt, alpha, channel_elasticity_D2,
            density, kinematic_viscosity)[1]
        ###############################################################################################################
        # iterative method to update the six unknowns at the conjunction point at next time step;
        # A1, Q1: cross-sectional area and flow rate at the outlet of parent vessel
        # A2, Q2: cross-sectional area and flow rate at the inlet of daughter1 vessel
        # A3, Q3: cross-sectional area and flow rate at the inlet of daughter2 vessel
        # Note: to find these six unknowns we need six equations

        # Eq. 1: Q1/A1 - W1_P + 4 √(ß_P / 2 ρ A0_P) * A1^0.25 = 0
        # W1_P at conjunction point (the outgoing characteristics of the parent vessel at the outlet of parent vessel)
        interpolation_function_P = interp1d(
            x(nb_nodes_P, tube_length_P), compute_W(A_old_P, Q_old_P, A0_P, density, channel_elasticity_P)[0],
            kind='linear', fill_value="extrapolate")
        W1_P = interpolation_function_P(
            tube_length_P - dt * (compute_lambda(A_old_P, Q_old_P, A0_P, density, channel_elasticity_P)[0])[-1])
        # Eq. 2 and Eq. 3 : Q2/A2 - W2_D1 - 4 √(ß_D1 / 2 ρ A0_D1) * A2^0.25 = 0
        #                   Q3/A3 - W2_D2 - 4 √(ß_D2 / 2 ρ A0_D2) * A3^0.25 = 0
        # W2_D1 and W2_D2  at conjunction point
        # the outgoing characteristics of the daughter vessels at the inlet of daughter vessels
        interpolation_function_D1 = interp1d(
             x(nb_nodes_D1, tube_length_D1), compute_W(A_old_D1, Q_old_D1, A0_D1, density, channel_elasticity_D1)[1],
             kind='linear', fill_value="extrapolate")
        interpolation_function_D2 = interp1d(
             x(nb_nodes_D2, tube_length_D2), compute_W(A_old_D2, Q_old_D2, A0_D2, density, channel_elasticity_D2)[1],
             kind='linear', fill_value="extrapolate")
        W2_D1 = interpolation_function_D1(
            -dt * (compute_lambda(A_old_D1, Q_old_D1, A0_D1, density, channel_elasticity_D1)[1])[0])
        W2_D2 = interpolation_function_D2(
            -dt * (compute_lambda(A_old_D2, Q_old_D2, A0_D2, density, channel_elasticity_D2)[1])[0])

        # Eq. 4
        # the conservation of mass across the bifurcation Q1 - Q2 - Q3 = 0

        # Eq. 5 and Eq. 6
        # the conservation of momentum flux across the bifurcation
        # 1/2 ρ (Q1/A1)^2 + P1  = 1/2 ρ (Q2/A2)^2 + P2
        # 1/2 ρ (Q1/A1)^2 + P1  = 1/2 ρ (Q3/A3)^2 + P3
        # P1, P2 and P3 can be replaced by the pressure-area relationship:
        # P-P0 = channel_elasticity/A0 (√A - √A0) ; P0 = 0
        # A modified Newton-Raphson method to find the six unknowns at the conjunction point at next time step
        # Define the system of equations
        def equations(unknowns):
            A1, Q1, A2, Q2, A3, Q3 = unknowns
            C1 = np.sqrt(channel_elasticity_P / (2 * density * A0_P))
            C2 = np.sqrt(channel_elasticity_D1 / (2 * density * A0_D1))
            C3 = np.sqrt(channel_elasticity_D2 / (2 * density * A0_D2))
            eq1 = (Q1 / A1) - W1_P + 4 * C1 * A1 ** 0.25
            eq2 = (Q2 / A2) - W2_D1 - 4 * C2 * A2 ** 0.25
            eq3 = (Q3 / A3) - W2_D2 - 4 * C3 * A3 ** 0.25
            eq4 = Q1 - Q2 - Q3
            eq5 = (
                    ((0.5 * density) * (Q1 / A1) ** 2) +
                    (channel_elasticity_P / A0_P) * (np.sqrt(A1) - np.sqrt(A0_P)) -
                    ((0.5 * density) * (Q2 / A2) ** 2) -
                    (channel_elasticity_D1 / A0_D1) * (np.sqrt(A2) - np.sqrt(A0_D1))
            )
            eq6 = (
                    ((0.5 * density) * (Q1 / A1) ** 2) +
                    (channel_elasticity_P / A0_P) * (np.sqrt(A1) - np.sqrt(A0_P)) -
                    ((0.5 * density) * (Q3 / A3) ** 2) -
                    (channel_elasticity_D2 / A0_D2) * (np.sqrt(A3) - np.sqrt(A0_D2))
            )
            return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

        # Initial guess for the six unknowns
        initial_guess = np.array([A_old_P[-1], Q_old_P[-1], A_old_D1[0], Q_old_D1[0], A_old_D2[0], Q_old_D2[0]])

        # Solve the system of equations using root
        solution = root(equations, initial_guess, method='lm', tol=1e-14)
        # print(n)
        # print(f"Did the solver converge? {'Yes' if solution.success else 'No'}")
        # print(f"Number of iterations (function evaluations): {solution.nfev}")
        A_P[-1] = solution.x[0]
        Q_P[-1] = solution.x[1]
        A_D1[0] = solution.x[2]
        Q_D1[0] = solution.x[3]
        A_D2[0] = solution.x[4]
        Q_D2[0] = solution.x[5]
        ###############################################################################################################
        # Now we need to define the cross-sectional area and flow rate at the
        #     1) inlet of parent vessel (A_P[0], Q_P[0] = inlet flow rate)
        #     2) outlet of daughter vessels (A_D1[-1], Q_D1[-1], A_D2[-1], Q_D2[-1])

        # 1) cross-sectional area and flow rate at the inlet of parent vessel
        Q_P[0] = Q0
        # W2_P_inlet and  W1_P_inlet
        interpolation_function_P_inlet = interp1d(
            x(nb_nodes_P, tube_length_P), compute_W(A_old_P, Q_old_P, A0_P, density, channel_elasticity_P)[1],
            kind='linear', fill_value="extrapolate")
        W2_P_inlet = interpolation_function_P_inlet(
            -dt * (compute_lambda(A_old_P, Q_old_P, A0_P, density, channel_elasticity_P)[1])[0])
        W1_P_inlet = -W2_P_inlet + 2 * (Q0 / A_old_P[0])
        A_P[0] = (((2 * density * A0_P) / channel_elasticity_P) ** 2) * ((W1_P_inlet - W2_P_inlet) / 8) ** 4
        # 2) cross-sectional area and flow rate at the inlet of parent vessel
        # W1_D1_outlet and  W1_D2_outlet
        # W1_D1_outlet
        interpolation_function_D1_outlet = interp1d(
             x(nb_nodes_D1, tube_length_D1), compute_W(A_old_D1, Q_old_D1, A0_D1, density, channel_elasticity_D1)[0],
             kind='linear', fill_value="extrapolate")
        W1_D1_outlet = interpolation_function_D1_outlet(
            tube_length_D1 - dt * (compute_lambda(A_old_D1, Q_old_D1, A0_D1, density, channel_elasticity_D1)[0])[-1]
        )
        # W1_D2_outlet
        interpolation_function_D2_outlet = interp1d(
             x(nb_nodes_D2, tube_length_D2), compute_W(A_old_D2, Q_old_D2, A0_D2, density, channel_elasticity_D2)[0],
             kind='linear', fill_value="extrapolate")
        W1_D2_outlet = interpolation_function_D2_outlet(
            tube_length_D2 - dt * (compute_lambda(A_old_D2, Q_old_D2, A0_D2, density, channel_elasticity_D2)[0])[-1]
        )

        A_D1[-1] = (((2 * density * A0_D1) / channel_elasticity_D1) ** 2) * ((W1_D1_outlet - W2_D1_outlet) / 8) ** 4
        Q_D1[-1] = (A_D1[-1] / 2) * (W1_D1_outlet + W2_D1_outlet)

        A_D2[-1] = (((2 * density * A0_D2) / channel_elasticity_D2) ** 2) * ((W1_D2_outlet - W2_D2_outlet) / 8) ** 4
        Q_D2[-1] = (A_D2[-1] / 2) * (W1_D2_outlet + W2_D2_outlet)
        ###############################################################################################################
        # final results for animation
        A_P_results.append(A_P.copy())
        A_D1_results.append(A_D1.copy())
        A_D2_results.append(A_D2.copy())
        ###############################################################################################################
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))  # show ax1 and ax2 figs
        ax1.set_xlim([0, tube_length_P])
        ax1.set_ylim([0.00005, 0.00017])
        ax2.set_xlim([0, tube_length_D1])
        ax2.set_ylim([3.5e-5, 5.2e-5])
        ax3.set_xlim([0, tube_length_D2])
        ax3.set_ylim([3.5e-5, 5.2e-5])

        line1, = ax1.plot(x(nb_nodes_P, tube_length_P), A_P_results[0], lw=2, label='Area_P (A)')
        line2, = ax2.plot(x(nb_nodes_D1, tube_length_D1), A_D1_results[0], lw=2, label='Area_D1 (A)')
        line3, = ax3.plot(x(nb_nodes_D2, tube_length_D2), A_D2_results[0], lw=2, label='Area_D2 (A)')
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

        def animate(i):
            line1.set_ydata(A_P_results[i])
            line2.set_ydata(A_D1_results[i])
            line3.set_ydata(A_D2_results[i])
            time_text.set_text(f'Time = {i * dt:.2f} s')
            return line1, line2, line3, time_text

        ani = FuncAnimation(fig, animate, frames=nt, interval=5, blit=True)
        ani.save("test_branching_inlet_flow_rate.gif")
        plt.xlabel('x')
        ax1.set_ylabel('Area_P (A)')
        ax2.set_ylabel('Area_D1 (A)')
        ax2.set_ylabel('Area_D2 (A)')

        ax1.set_title('Variation of A_P with respect to x_P')
        ax2.set_title('Variation of A_D1 with respect to x_D1')
        ax3.set_title('Variation of A_D2 with respect to x_D2')
        plt.tight_layout()
        plt.show()

import numpy as np
import scipy.sparse


def area_change(positions_n, area_e, dt, flow_rate):
    """
    Calculates the time change rate of area across a pipe

    Parameters
    ----------
    positions_n : np.ndarray
        The nodal positions along the channel.
    area_e : float
        The area at each element in the past time step.
    dt : float
        The time step
    flow_rate : float
        The flow rate at each node in the past time step.
    Returns
    -------
    area_e_new : np.ndarray
        the new area at each element along the channel.
    """
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]
    area_e = np.asanyarray(area_e)
    flow_rate = np.asanyarray(flow_rate)
    flow_rate[0] = 4e-7  # Dirichlet boundary condition
    return area_e + (dt * ((flow_rate[:-1] - flow_rate[1:]) / element_lengths_e[:]))


def flow_rate_change(positions_n, area_e, dt, flow_rate, pressure, density, kinematic_viscosity, alpha=4 / 3):
    """
        Calculates the time change rate of the flow rate

        Parameters
        ----------
        positions_n : np.ndarray
            The nodal positions along the channel.
        area_e : float
            The area at each element in the past time step.
        dt : float
            The time step
        flow_rate : float
            The flow rate at each node in the past time step
        pressure : float
            The pressure at each node in the past time
        density : float
            The fluid density
        kinematic_viscosity : float
            The fluid kinematic viscosity
        alpha : float
            Momentum correction factor. (Default: 4/3)
        Returns
        -------
        flow_rate_n : np.ndarray
            the time change rate of area at each element along the channel.
        """
    nb_nodes = len(positions_n)
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]
    area_e = np.asanyarray(area_e)

    # The matrix form of this system is given by M (dQ/dt) = -C -K(p) -VQ

    # Diagonal entries of the consistent mass matrix M
    diagonal_entries_m = np.ones(nb_nodes)
    diagonal_entries_m[1:-1] = (2 * (element_lengths_e[:-1] + element_lengths_e[1:])) / 6
    diagonal_entries_m[-1] = (2 * (element_lengths_e[-1])) / 6
    # Off-diagonals entries of the consistent mass matrix M
    off_diagonal_entries_upper_m = np.copy((element_lengths_e[:]) / 6)
    off_diagonal_entries_upper_m[0] = 0  # Dirichlet boundary condition
    off_diagonal_entries_lower_m = np.copy((element_lengths_e[:]) / 6)
    # We need to append 0 here to make the array the same length as diagonal_entries_m
    off_diagonal_entries_upper_m = np.append(0, off_diagonal_entries_upper_m)
    off_diagonal_entries_lower_m = np.append(off_diagonal_entries_lower_m, 0)
    # Construct the consistent mass matrix M
    M = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_m,
                                           diagonal_entries_m,
                                           off_diagonal_entries_upper_m]),
                                 np.array([-1, 0, 1])), shape=(nb_nodes, nb_nodes))
    # Construct the nonlinear convective vector C
    C = np.zeros(nb_nodes)
    C[0] = (alpha/3) * (((-2 * (flow_rate[0] ** 2)) + (flow_rate[0] * flow_rate[1]) + (flow_rate[1] ** 2)) / area_e[0])
    C[-1] = (alpha/3) * (((-flow_rate[-2] ** 2) - (flow_rate[-2] * flow_rate[-1]) + (2 * (flow_rate[-1] ** 2))) / area_e[-1])
    C[1:-1] = (alpha/3) * ((((-flow_rate[:-2] ** 2) - (flow_rate[:-2] * flow_rate[1:-1]) + (2 * (flow_rate[1:-1] ** 2))) / area_e[:-1]) + (((-2*(flow_rate[1:-1]**2)) + (flow_rate[1:-1]*flow_rate[2:]) + (flow_rate[2:]**2)) / area_e[1:]))
    # Construct the viscose matrix V
    # Diagonal entries of the viscose matrix V
    diagonal_entries_v = np.ones(nb_nodes)
    diagonal_entries_v[1:-1] = (2*alpha*kinematic_viscosity*np.pi) * ((element_lengths_e[:-1] / area_e[:-1]) + (element_lengths_e[1:] / area_e[1:]))
    diagonal_entries_v[0] = (2*alpha*kinematic_viscosity*np.pi) * (element_lengths_e[0] / area_e[0])
    diagonal_entries_v[-1] = (2*alpha*kinematic_viscosity*np.pi) * (element_lengths_e[-1] / area_e[-1])
    # Off-diagonals entries of the viscose matrix V
    off_diagonal_entries_upper_v = element_lengths_e[:] / area_e[:]
    off_diagonal_entries_lower_v = element_lengths_e[:]/ area_e[:]
    # We need to append 0 here to make the array the same length as diagonal_entries_v
    off_diagonal_entries_upper_v = np.append(0, off_diagonal_entries_upper_v)
    off_diagonal_entries_lower_v = np.append(off_diagonal_entries_lower_v, 0)
    # Construct the viscose matrix V
    V = scipy.sparse.dia_matrix((np.array(
        [off_diagonal_entries_lower_v, diagonal_entries_v, off_diagonal_entries_upper_v]), np.array([-1, 0, 1])),
                                shape=(nb_nodes, nb_nodes))
    # Diagonal entries of the K matrix
    diagonal_entries_k = np.ones(nb_nodes)  # First entry needs to be unity (Dirichlet boundary condition)
    diagonal_entries_k[1:-1] = (area_e[:-1] - area_e[1:]) / (2 * density)
    diagonal_entries_k[0] = area_e[0] / (-2 * density)
    diagonal_entries_k[-1] = area_e[-1] / (2 * density)  # Last entry (Neumann boundary condition)
    # Off-diagonals entries of the K matrix
    off_diagonal_entries_upper_k = area_e[:] / (2 * density)
    off_diagonal_entries_lower_k = -area_e[:] / (2 * density)

    # We need to append 0 here to make the array the same length as diagonal_entries_n
    off_diagonal_entries_upper_k = np.append(0, off_diagonal_entries_upper_k)
    off_diagonal_entries_lower_k = np.append(off_diagonal_entries_lower_k, 0)
    # Construct the K matrix
    K = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_k,
                                           diagonal_entries_k,
                                           off_diagonal_entries_upper_k]),
                                 np.array([-1, 0, 1])), shape=(nb_nodes, nb_nodes))
    # Right hand side
    rhs = (M * flow_rate) - (dt*C) - (dt*K*pressure) - (dt*V*flow_rate)
    rhs[0] = 4e-7  # Dirichlet boundary condition
    return scipy.sparse.linalg.spsolve(M, rhs)

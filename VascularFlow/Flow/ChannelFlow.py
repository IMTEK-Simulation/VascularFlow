import numpy as np
import scipy.sparse


def flow_rate(position_n, area_rate_of_change_e, input_flow_rate):
    """
    Calculates the mass conservation of a fluid in a channel using linear
    finite elements or (equivalently) first-order finite differences.

    Parameters
    ----------
    position_n : np.ndarray
        The nodal positions along the channel.
    area_rate_of_change_e : np.ndarray
        The time-rate of change of the channel cross sectional area
        within each element along the channel position.
    input_flow_rate : float
        The flow rate at the input of the channel.

    Returns
    -------
    flow_rate_n : np.ndarray
        The flow rate at each nodal position along the channel.
    """
    element_lengths_e = position_n[1:] - position_n[:-1]
    return np.append([input_flow_rate],
                     input_flow_rate + np.cumsum(area_rate_of_change_e * element_lengths_e))


def pressure_change(positions_n, area_e, flow_rate, density, kinematic_viscosity, alpha=4 / 3):
    """
    Calculates the pressure drop across a pipe of given radius and length
    for a given flow rate and viscosity.

    Parameters
    ----------
    position_n : np.ndarray
        The nodal positions along the channel.
    area_e : np.ndarray
        The cross-sectional area of the pipe for each element along the pipe.
    flow_rate : float
        The flow rate.
    density : float
        The density of the fluid.
    kinematic_viscosity : float
        The kinematic viscosity of the fluid.
    alpha : float
        Momentum correction factor. (Default: 4/3)

    Returns
    -------
    pressure_change_n : np.ndarray
        The pressure change across the pipe for each nodal position along the
        pipe.
    """
    positions_n = np.asanyarray(positions_n)
    area_e = np.asanyarray(area_e)

    nb_nodes = len(positions_n)

    # Diagonal entries of the stiffness matrix
    diagonal_entries_n = np.ones(nb_nodes)  # First entry needs to be unity (Dirichlet boundary condition)
    diagonal_entries_n[1:-1] = (area_e[:-1] - area_e[1:]) / (2 * density)
    diagonal_entries_n[-1] = area_e[-1] / (2 * density)  # Last entry (Neumann boundary condition)
    # Off-diagonals entries of the stiffness matrix
    off_diagonal_entries_upper_e = area_e / (2 * density)
    off_diagonal_entries_lower_e = -area_e / (2 * density)
    off_diagonal_entries_upper_e[0] = 0  # Dirichlet boundary condition

    # We need to append 0 here to make the array the same length as diagonal_entries_n
    off_diagonal_entries_upper_e = np.append(0, off_diagonal_entries_upper_e )
    off_diagonal_entries_lower_e = np.append(off_diagonal_entries_lower_e, 0)
    # Construct stiffness matrix
    K = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_e,
                                           diagonal_entries_n,
                                           off_diagonal_entries_upper_e]),
                                 np.array([-1, 0, 1])), shape=(nb_nodes, nb_nodes))

    # Right hand side vector
    f = np.zeros(nb_nodes)
    dx_e = positions_n[1:] - positions_n[:-1]
    f[1:-1] = -4 * np.pi * kinematic_viscosity * flow_rate * (dx_e[:-1] / area_e[:-1] + dx_e[1:] / area_e[1:]) + \
              alpha * flow_rate ** 2 * (1 / area_e[:-1] - 1 / area_e[1:])

    # Set boundary terms
    f[0] = 0
    f[-1] = -4 * np.pi * kinematic_viscosity * flow_rate * dx_e[-1] / area_e[-1]

    return scipy.sparse.linalg.spsolve(K, f)

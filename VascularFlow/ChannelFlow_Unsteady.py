import numpy as np
import scipy.sparse


def dAdt(positions_n, flow_rate_n):
    """
    Calculates the time rate of area changes for each element across the channel.

    Parameters
    ----------
    positions_n : np.ndarray
        The nodal positions along the channel.
    flow_rate_n : np.ndarray
        The flow rate at each node along the channel.
    Returns
    -------
        the time rate of area changes for each element across the channel.
    """
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]
    return (flow_rate_n[:-1] - flow_rate_n[1:]) / (element_lengths_e[:])


def new_area_e(area_e, dt, positions_n, flow_rate_n):
    """
    Calculates the cross-sectional area for each element across the channel at the next time step.

    Parameters
    ----------
    area_e : np.ndarray
        the cross-sectional area for each element across the channel at the previous time step.
    dt : float
        time step.
    positions_n : np.ndarray
        the nodal positions along the channel
    flow_rate_n : np.ndarray
        The flow rate at each node along the
    Returns
    -------
    time_changes_area_e: np.ndarray
        the time rate of area changes for each element across the channel.
    """
    return (dAdt(positions_n, flow_rate_n)*dt) + area_e


def dQdt(positions_n, area_e, flow_rate_n, A0, density, kinematic_viscosity, ring_modulus, alpha=4 / 3):
    """
    Calculates the time rate of flow rate changes for each node across the channel.

    Parameters
    ----------
    positions_n : np.ndarray
        The nodal positions along the channel.
    area_e : np.ndarray
        the cross-sectional area for each element across the channel.
    flow_rate_n : np.ndarray
        The flow rate at each node along the channel.
    A0 : float
        the area of the unstressed channel.
    density : float
        The fluid density
    kinematic_viscosity : float
        The fluid kinematic viscosity
    ring_modulus : float
        The ring modulus of the tube.
    alpha : float
        Momentum correction factor. (Default: 4/3)
    Returns
    -------
        the time rate of flow rate changes for each node across the channel.
    """
    nb_nodes = len(positions_n)
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]

    # The matrix form of this system is given by M (dQ/dt) = C + N - VQ

    # Diagonal entries of M matrix
    diagonal_entries_m = np.ones(nb_nodes)
    diagonal_entries_m[1:-1] = (2 * (element_lengths_e[:-1] + element_lengths_e[1:])) / 6
    diagonal_entries_m[0] = (2 * (element_lengths_e[0])) / 6
    diagonal_entries_m[-1] = (2 * (element_lengths_e[-1])) / 6
    # Off-diagonals entries of M matrix
    off_diagonal_entries_upper_m = np.copy((element_lengths_e[:]) / 6)
    off_diagonal_entries_lower_m = np.copy((element_lengths_e[:]) / 6)
    # We need to append 0 here to make the array the same length as diagonal_entries_m
    off_diagonal_entries_upper_m = np.append(0, off_diagonal_entries_upper_m)
    off_diagonal_entries_lower_m = np.append(off_diagonal_entries_lower_m, 0)
    # Construct M matrix
    M = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_m, diagonal_entries_m,
                                           off_diagonal_entries_upper_m]),
                                 np.array([-1, 0, 1])), shape=(nb_nodes, nb_nodes))
    # Construct C matrix
    C = np.zeros(nb_nodes)
    C[0] = (alpha / 3) * (-(flow_rate_n[0] ** 2) - (flow_rate_n[0] * flow_rate_n[1]) - (flow_rate_n[1] ** 2)) / (area_e[0])
    C[-1] = (alpha / 3) * ((flow_rate_n[-2] ** 2) + (flow_rate_n[-2] * flow_rate_n[-1]) + (flow_rate_n[-1] ** 2)) / (area_e[-1])
    C[1:-1] = (alpha / 3) * ((((flow_rate_n[:-2] ** 2) + (flow_rate_n[:-2] * flow_rate_n[1:-1]) + (flow_rate_n[1:-1] ** 2)) / (area_e[:-1])) - (((flow_rate_n[1:-1] ** 2) + (flow_rate_n[1:-1] * flow_rate_n[2:]) + (flow_rate_n[2:] ** 2)) / (area_e[1:])))
    # Construct N matrix
    N = np.zeros(nb_nodes)
    N[0] = (ring_modulus / (2 * density * A0)) * (-1 * (area_e[0] ** 2))
    N[-1] = (ring_modulus / (2 * density * A0)) * (area_e[-1] ** 2)
    N[1:-1] = (ring_modulus / (2 * density * A0)) * ((area_e[:-1] ** 2) - (area_e[1:] ** 2))
    # Construct V matrix
    # Diagonal entries of V matrix
    diagonal_entries_v = np.ones(nb_nodes)
    diagonal_entries_v[1:-1] = (2 * alpha * kinematic_viscosity * np.pi) * (
                (element_lengths_e[:-1] / area_e[:-1]) + (element_lengths_e[1:] / area_e[1:]))
    diagonal_entries_v[0] = (2 * alpha * kinematic_viscosity * np.pi) * (element_lengths_e[0] / area_e[0])
    diagonal_entries_v[-1] = (2 * alpha * kinematic_viscosity * np.pi) * (element_lengths_e[-1] / area_e[-1])
    # Off-diagonals entries of V matrix
    off_diagonal_entries_upper_v = (alpha * kinematic_viscosity * np.pi) * (element_lengths_e[:] / area_e[:])
    off_diagonal_entries_lower_v = (alpha * kinematic_viscosity * np.pi) * (element_lengths_e[:] / area_e[:])
    # We need to append 0 here to make the array the same length as diagonal_entries_v
    off_diagonal_entries_upper_v = np.append(0, off_diagonal_entries_upper_v)
    off_diagonal_entries_lower_v = np.append(off_diagonal_entries_lower_v, 0)
    V = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_v, diagonal_entries_v,
                                           off_diagonal_entries_upper_v]),
                                 np.array([-1, 0, 1])),
                                shape=(nb_nodes, nb_nodes))
    # Right hand side
    rhs = C + N - (V * flow_rate_n)
    return scipy.sparse.linalg.spsolve(M, rhs)


def new_flow_rate_n(positions_n, area_e, flow_rate_n, A0, dt, density, kinematic_viscosity, ring_modulus):
    """
    Calculates the flow rate for each node across the channel at the next time step.

    Parameters
    ----------
    positions_n : np.ndarray
        The nodal positions along the channel.
    area_e : np.ndarray
        the cross-sectional area for each element across the channel.
    flow_rate_n : np.ndarray
        the flow rate for each node across the channel at the previous time step.
    A0 : float
        the area of the unstressed channel.
    dt : float
        time step.
    density : float
        The fluid density
    kinematic_viscosity : float
        The fluid kinematic viscosity
    ring_modulus : float
        The ring modulus of the tube.
    Returns
    -------
        the flow rate for each node across the channel at the next time step.
    """
    return ((dQdt(positions_n, area_e, flow_rate_n, A0, density, kinematic_viscosity, ring_modulus, alpha=4 / 3)*dt) +
            flow_rate_n)

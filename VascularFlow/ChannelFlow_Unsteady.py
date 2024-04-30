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
    nb_nodes = positions_n.size
    nb_elements = nb_nodes - 1
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]
    # M1
    diagonal_entries_m1 = np.ones(nb_elements)
    diagonal_entries_m1[1:] = element_lengths_e[1:]
    off_diagonal_entries_upper_m1 = np.zeros(nb_elements - 1)
    off_diagonal_entries_lower_m1 = element_lengths_e[:-1]
    off_diagonal_entries_upper_m1 = np.append(0, off_diagonal_entries_upper_m1)
    off_diagonal_entries_lower_m1 = np.append(off_diagonal_entries_lower_m1, 0)
    M1 = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_m1, diagonal_entries_m1,
                                            off_diagonal_entries_upper_m1]),
                                  np.array([-1, 0, 1])), shape=(nb_elements, nb_elements))
    # V1
    V1 = np.zeros(nb_elements)
    V1[1:] = flow_rate_n[:-2] - flow_rate_n[2:]
    return scipy.sparse.linalg.spsolve(M1, V1)


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
    nb_nodes = positions_n.size
    nb_elements = nb_nodes - 1
    positions_n = np.asanyarray(positions_n)
    element_lengths_e = positions_n[1:] - positions_n[:-1]
    # M
    diagonal_entries_m = np.ones(nb_nodes)
    diagonal_entries_m[1:-1] = (element_lengths_e[:-1] + element_lengths_e[1:]) / 3
    diagonal_entries_m[-1] = element_lengths_e[-1] / 3
    off_diagonal_entries_upper_m = np.zeros(nb_elements)
    off_diagonal_entries_upper_m[1:] = np.copy(element_lengths_e[1:] / 6)
    off_diagonal_entries_lower_m = np.copy(element_lengths_e / 6)
    off_diagonal_entries_upper_m = np.append(0, off_diagonal_entries_upper_m)
    off_diagonal_entries_lower_m = np.append(off_diagonal_entries_lower_m, 0)
    M = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_m, diagonal_entries_m,
                                           off_diagonal_entries_upper_m]),
                                 np.array([-1, 0, 1])), shape=(nb_nodes, nb_nodes))
    # C
    C = np.zeros(nb_nodes)
    C[0] = (alpha / 3) * (-(flow_rate_n[0] ** 2) - (flow_rate_n[0] * flow_rate_n[1]) - (flow_rate_n[1] ** 2)) / (
    area_e[0])
    C[-1] = (alpha / 3) * ((flow_rate_n[-2] ** 2) + (flow_rate_n[-2] * flow_rate_n[-1]) + (flow_rate_n[-1] ** 2)) / (
    area_e[-1])
    C[1:-1] = (alpha / 3) * ((((flow_rate_n[:-2] ** 2) + (flow_rate_n[:-2] * flow_rate_n[1:-1]) +
                               (flow_rate_n[1:-1] ** 2)) /
                              (area_e[:-1])) - (((flow_rate_n[1:-1] ** 2) + (flow_rate_n[1:-1] * flow_rate_n[2:]) +
                                                 (flow_rate_n[2:] ** 2)) / (area_e[1:])))
    # N
    N = np.zeros(nb_nodes)
    N[0] = (ring_modulus / (2 * density * A0)) * (-1 * (area_e[0] ** 2))
    N[-1] = (ring_modulus / (2 * density * A0)) * ((area_e[-1] ** 2))
    N[1:-1] = (ring_modulus / (2 * density * A0)) * ((area_e[:-1] ** 2) - (area_e[1:] ** 2))

    # V
    diagonal_entries_v = np.ones(nb_nodes)
    diagonal_entries_v[1:-1] = (2 * alpha * kinematic_viscosity * np.pi) * (
                (element_lengths_e[:-1] / area_e[:-1]) + (element_lengths_e[1:] / area_e[1:]))
    diagonal_entries_v[0] = (2 * alpha * kinematic_viscosity * np.pi) * (element_lengths_e[0] / area_e[0])
    diagonal_entries_v[-1] = (2 * alpha * kinematic_viscosity * np.pi) * (element_lengths_e[-1] / area_e[-1])
    off_diagonal_entries_upper_v = (alpha * kinematic_viscosity * np.pi) * (element_lengths_e[:] / area_e[:])
    off_diagonal_entries_lower_v = (alpha * kinematic_viscosity * np.pi) * (element_lengths_e[:] / area_e[:])
    off_diagonal_entries_upper_v = np.append(0, off_diagonal_entries_upper_v)
    off_diagonal_entries_lower_v = np.append(off_diagonal_entries_lower_v, 0)
    V = scipy.sparse.dia_matrix((np.array([off_diagonal_entries_lower_v, diagonal_entries_v,
                                           off_diagonal_entries_upper_v]),
                                 np.array([-1, 0, 1])),
                                shape=(nb_nodes, nb_nodes))
    rhs = C + N - (V * flow_rate_n)
    rhs[0] = 0
    return scipy.sparse.linalg.spsolve(M, rhs)




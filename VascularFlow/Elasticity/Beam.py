import numpy as np



def euler_bernoulli(x_n, p_n):
    """
    Calculates the deflection of a beam under the Euler-Bernoulli beam theory.

    Parameters
    ----------
    x_n : np.ndarray
        The nodal positions along the beam.
    p_n : np.ndarray
        The pressure across the beam for each nodal position along the beam.

    Returns
    -------
    deflection_n : np.ndarray
        The deflection of the beam for each nodal position along the beam.
    """
    x_n = np.asanyarray(x_n)
    p_n = np.asanyarray(p_n)

    nb_nodes = len(x_n)
    element_lengths_e = x_n[1:] - x_n[:-1]
    element_centers_e = (x_n[1:] + x_n[:-1]) / 2

    deflection_n = np.zeros(nb_nodes)
    for i in range(1, nb_nodes):
        deflection_n[i] = deflection_n[i-1] + (p_n[i-1] * element_lengths_e[i-1]**3) / 3
    return deflection_n
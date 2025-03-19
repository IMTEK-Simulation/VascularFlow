import numpy as np

from VascularFlow.Numerics.Assembly import assemble_system_matrix
from VascularFlow.Numerics.BasisFunctions import QuadraticBasis
from VascularFlow.Numerics.ElementMatrices import second_second


def euler_bernoulli(x_n, q_g):
    """
    Calculates the deflection of a beam under the Euler-Bernoulli beam theory.

    Parameters
    ----------
    x_n : np.ndarray
        The positions of the element boundary along the beam.
    q_g : np.ndarray
        The line-load across the beam for each nodal position along the beam.
        (The line-load is normalized by EI)

    Returns
    -------
    deflection_n : np.ndarray
        The deflection of the beam for each nodal position along the beam.
    """
    element_matrix_nn = second_second(1, QuadraticBasis())
    nb_nodes, _ = element_matrix_nn.shape

    dx_e = x_n[1:] - x_n[:-1]  # Width of each element
    element_matrices_enn = element_matrix_nn.reshape((1, nb_nodes, nb_nodes)) * (
        dx_e ** (-3)
    ).reshape(-1, 1, 1)

    system_matrix_gg = assemble_system_matrix(element_matrices_enn)

    w_g = np.linalg.solve(system_matrix_gg, q_g)

    return w_g

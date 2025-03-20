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
    n, _ = element_matrix_nn.shape

    # Compute element matrices
    dx_e = x_n[1:] - x_n[:-1]  # Width of each element
    element_matrices_enn = element_matrix_nn.reshape((1, n, n)) * (
        dx_e ** (-3)
    ).reshape(-1, 1, 1)

    # Assemble system matrix
    system_matrix_gg = assemble_system_matrix(element_matrices_enn)

    # Add boundary conditions
    # Fixed displacement left
    system_matrix_gg[0] = 0
    system_matrix_gg[0, 0] = 1
    q_g[0] = 0
    # Fixed derivative left
    system_matrix_gg[1, 0] = -3
    system_matrix_gg[1, 1] = 4
    system_matrix_gg[1, 2] = -1
    q_g[1] = 0
    # Fixed displacement right
    system_matrix_gg[-1] = 0
    system_matrix_gg[-1, -1] = 1
    q_g[-1] = 0
    # Fixed derivative right
    system_matrix_gg[-2, -1] = -3
    system_matrix_gg[-2, -2] = 4
    system_matrix_gg[-2, -3] = -1
    q_g[-2] = 0

    # Solve system
    w_g = np.linalg.solve(system_matrix_gg, q_g)

    return w_g

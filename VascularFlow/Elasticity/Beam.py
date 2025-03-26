import numpy as np

from VascularFlow.Numerics.Assembly import (
    assemble_system_matrix_2dof,
    assemble_force_matrix_2dof,
)
from VascularFlow.Numerics.BasisFunctions import HermiteBasis
from VascularFlow.Numerics.ElementMatrices import (
    second_second,
    force_matrix,
    mass_matrix,
)


def euler_bernoulli(x_n, dx_e, p):
    """
    Calculates the deflection of a beam under the Euler-Bernoulli beam theory.

    Parameters
    ----------
    x_n : np.ndarray
        The positions of the element boundary along the beam.
    dx_e : np.ndarray
        The element length of the beam.
    q_g : np.ndarray
        The line-load across the beam for each nodal position along the beam.
        (The line-load is normalized by EI)

    Returns
    -------
    deflection_n : np.ndarray
        The deflection of the beam for each nodal position along the beam.
    """
    element_matrix_nn = second_second(3, dx_e, HermiteBasis())
    nb_nodes, _ = element_matrix_nn.shape
    nb_elements = len(x_n) - 1
    element_matrices_enn = element_matrix_nn.reshape((1, nb_nodes, nb_nodes)) * np.ones(
        nb_elements
    ).reshape(nb_elements, 1, 1)
    system_matrix_gg = assemble_system_matrix_2dof(element_matrices_enn)

    rhs1 = force_matrix(dx_e)
    rhs2 = rhs1.reshape((1, 1, 4)) * np.ones(nb_elements).reshape(nb_elements, 1, 1)
    system_matrix_ll = assemble_force_matrix_2dof(rhs2) * p

    # Add boundary conditions

    system_matrix_gg[0] = 0
    system_matrix_gg[0, 0] = 1
    system_matrix_gg[1] = 0
    system_matrix_gg[1, 1] = 1
    system_matrix_gg[-1] = 0
    system_matrix_gg[-1, -1] = 1
    system_matrix_gg[-2] = 0
    system_matrix_gg[-2, -2] = 1

    system_matrix_ll[0] = 0
    system_matrix_ll[1] = 0
    system_matrix_ll[-1] = 0
    system_matrix_ll[-2] = 0

    # Solve system
    w_g = np.linalg.solve(system_matrix_gg, system_matrix_ll)

    return system_matrix_gg, system_matrix_ll, w_g


def euler_bernoulli_transient(x_n, dx_e, num_steps, dt, p, beta, relaxation, H_new):
    element_matrix_stiffness = second_second(3, dx_e, HermiteBasis())
    nb_nodes, _ = element_matrix_stiffness.shape
    nb_elements = len(x_n) - 1
    assemble_matrix_stiffness = element_matrix_stiffness.reshape(
        (1, nb_nodes, nb_nodes)
    ) * np.ones(nb_elements).reshape(nb_elements, 1, 1)
    k = assemble_system_matrix_2dof(assemble_matrix_stiffness)

    element_matrix_mass = mass_matrix(3, dx_e, HermiteBasis())
    nb_nodes, _ = element_matrix_mass.shape
    assemble_matrix_mass = element_matrix_mass.reshape(
        (1, nb_nodes, nb_nodes)
    ) * np.ones(nb_elements).reshape(nb_elements, 1, 1)
    m = assemble_system_matrix_2dof(assemble_matrix_mass)

    a = m + (dt**2) * k

    element_load_vector = force_matrix(dx_e)
    assemble_load_vector = element_load_vector.reshape((1, 1, 4)) * np.ones(
        nb_elements
    ).reshape(nb_elements, 1, 1)
    p_interleaved = np.zeros(len(p) * 2)
    p_interleaved[::2] = p
    f = assemble_force_matrix_2dof(assemble_load_vector) * p_interleaved

    a[0] = 0
    a[0, 0] = 1
    a[1] = 0
    a[1, 1] = 1
    a[-1] = 0
    a[-1, -1] = 1
    a[-2] = 0
    a[-2, -2] = 1

    w_n = np.zeros(2 * len(x_n))
    w_n1 = np.zeros(2 * len(x_n))
    for n in range(num_steps):

        r1 = 2 * (m @ w_n)
        r2 = m @ w_n1
        r3 = (dt**2) * f
        q = r1 - r2 + r3
        q[0] = 0
        q[1] = 0
        q[-1] = 0
        q[-2] = 0
        w_g = np.linalg.solve(a, q)

        w_n1 = w_n
        w_n = w_g

    H_new_interleaved = np.zeros(len(H_new) * 2)
    H_new_interleaved[::2] = H_new
    H_g = 1 + (beta * w_g)
    H_g = relaxation * H_g + (1 - relaxation) * H_new_interleaved

    return H_g[::2]

import numpy as np

from VascularFlow.Numerics.Assembly import assemble_system_matrix_1dof, assemble_force_matrix_pressure
from VascularFlow.Numerics.BasisFunctions import LinearBasis
from VascularFlow.Numerics.ElementMatrices import eval_first, force_matrix_pressure
from VascularFlow.Numerics.ArrayFirstDerivative import array_first_derivative


def pressure(x_n, dx_e, dt, eps, re, st, Hstar, Qstar, Q_n, Q_n1):
    element_matrix_nn = eval_first(1, LinearBasis())
    nb_nodes, _ = element_matrix_nn.shape
    nb_elements = len(x_n) - 1
    element_matrices_enn = element_matrix_nn.reshape((1, nb_nodes, nb_nodes)) * np.ones(nb_elements).reshape(
        nb_elements, 1, 1)
    system_matrix_gg = assemble_system_matrix_1dof(element_matrices_enn)
    system_matrix_gg[-1] = 0
    system_matrix_gg[-1, -1] = 1

    first_term = -((eps * re * st) / Hstar) * (3 * Qstar - 4 * Q_n + Q_n1) / (2 * dt)
    second_term = 1.2 * re * (1 / Hstar) * array_first_derivative(Qstar**2/Hstar, dx_e)
    third_term = -12 * Qstar / (Hstar ** 3)

    rhs1 = force_matrix_pressure(dx_e)
    rhs2 = rhs1.reshape((1, 1, 2)) * np.ones(nb_elements).reshape(nb_elements, 1, 1)
    system_matrix_ll = assemble_force_matrix_pressure(rhs2) * (first_term + second_term + third_term)

    system_matrix_ll[-1] = 0

    p = np.linalg.solve(system_matrix_gg, system_matrix_ll)

    return system_matrix_gg, system_matrix_ll , p

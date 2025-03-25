import numpy as np

from VascularFlow.Numerics.BasisFunctions import LinearBasis
from VascularFlow.Numerics.ElementMatrices import eval_first, force_matrix_pressure
from VascularFlow.Numerics.Assembly import assemble_system_matrix_1dof , assemble_force_matrix_pressure


def flow_rate(x_n, dx_e, dt, st, inlet_flow_rate, Hstar, H_n, H_n1):
    element_matrix_nn = eval_first(1, LinearBasis())
    nb_nodes, _ = element_matrix_nn.shape
    nb_elements = len(x_n) - 1
    element_matrices_enn = element_matrix_nn.reshape((1, nb_nodes, nb_nodes)) * np.ones(nb_elements).reshape(
        nb_elements, 1, 1)
    system_matrix_gg = assemble_system_matrix_1dof(element_matrices_enn)
    system_matrix_gg[0] = 0
    system_matrix_gg[0, 0] = 1

    first_term = (-st * (3 * Hstar - 4 * H_n + H_n1)) / (2 * dt)


    rhs1 = force_matrix_pressure(dx_e)
    rhs2 = rhs1.reshape((1, 1, 2)) * np.ones(nb_elements).reshape(nb_elements, 1, 1)
    system_matrix_ll = assemble_force_matrix_pressure(rhs2) * first_term

    system_matrix_ll[0] = inlet_flow_rate

    q = np.linalg.solve(system_matrix_gg, system_matrix_ll)

    return q
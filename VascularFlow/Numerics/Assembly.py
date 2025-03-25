import numpy as np
from numpy import ndarray


def assemble_system_matrix_1dof(element_matrices_enn: ndarray):
    """Assemble the system matrix from the element matrices."""
    nb_elements, nb_nodes, _ = element_matrices_enn.shape
    assert element_matrices_enn.shape == (nb_elements, nb_nodes, nb_nodes)
    nb_global_nodes = (nb_nodes - 1) * nb_elements + 1

    system_matrix_gg = np.zeros((nb_global_nodes, nb_global_nodes))

    for e in range(nb_elements):
        system_matrix_gg[
            e * (nb_nodes - 1) : (e + 1) * (nb_nodes - 1) + 1,
            e * (nb_nodes - 1) : (e + 1) * (nb_nodes - 1) + 1,
        ] += element_matrices_enn[e]

    return system_matrix_gg


def assemble_system_matrix_2dof(element_matrices_enn: ndarray):
    """Assemble the system matrix from the element matrices."""
    nb_elements, nb_nodes, _ = element_matrices_enn.shape
    assert element_matrices_enn.shape == (nb_elements, nb_nodes, nb_nodes)
    nb_global_nodes = nb_elements * 2 + 2

    system_matrix_gg = np.zeros((nb_global_nodes, nb_global_nodes))

    for e in range(nb_elements):
        start = e * (nb_nodes - 2)
        end = start + nb_nodes
        system_matrix_gg[
            start : end,
            start : end,
        ] += element_matrices_enn[e]
    return system_matrix_gg

def assemble_force_matrix_2dof(element_matrices_enn: ndarray):
    """Assemble the system matrix from the element matrices."""
    nb_elements, _, nb_nodes = element_matrices_enn.shape
    nb_global_nodes = nb_elements * 2 + 2

    system_matrix_gg = np.zeros(nb_global_nodes)

    for e in range(nb_elements):
        start = e * (nb_nodes - 2)
        end = start + nb_nodes
        system_matrix_gg[start:end] += element_matrices_enn[e].reshape(4,)
    return system_matrix_gg


def assemble_force_matrix_pressure(element_matrices_enn: ndarray):
    """Assemble the system matrix from the element matrices."""
    nb_elements, _, nb_nodes = element_matrices_enn.shape
    nb_global_nodes = nb_elements + 1
    system_matrix_gg = np.zeros(nb_global_nodes)

    for e in range(nb_elements):
        start = e * (nb_nodes - 1)
        end = start + nb_nodes
        system_matrix_gg[start:end] += element_matrices_enn[e].reshape(2,)
    return system_matrix_gg
from numpy import ndarray


def assemble_system_matrix(element_matrices_enn: ndarray):
    """Assemble the system matrix from the element matrices."""
    nb_elements, nb_nodes, _ = element_matrices_enn.shape
    assert element_matrices_enn.shape == (nb_elements, nb_nodes, nb_nodes)

    nb_global_nodes = (nb_nodes - 1) * nb_elements + 1

    system_matrix_gg = ndarray((nb_global_nodes, nb_global_nodes))

    for e in range(nb_elements):
        system_matrix_gg[
            e * (nb_nodes - 1) : (e + 1) * (nb_nodes - 1) + 1,
            e * (nb_nodes - 1) : (e + 1) * (nb_nodes - 1) + 1,
        ] += element_matrices_enn[e]

    return system_matrix_gg

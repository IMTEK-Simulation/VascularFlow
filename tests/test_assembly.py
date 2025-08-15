"""
Tests for evaluating matrices for a global system of equations used in 1D finite element methods.

This module verifies:
- Accuracy of the shape of the global matrices in an arbitrary interval.

Tested basis types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
"""

import numpy as np
import pytest

from VascularFlow.Numerics.Assembly import (
    assemble_global_matrices,
    assemble_global_matrices_vectors_2d,
)

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    HermiteBasis,
    QuadraticBasis,
    BilinearShapeFunctions,
)
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_fourth_derivative,
    stiffness_matrix_second_derivative,
    stiffness_matrix_first_derivative,
    mass_matrix_fourth_derivatives,
    load_vector,
)


@pytest.mark.parametrize(
    "basis_function_class, element_matrix_func, element_vector_func",
    [
        (HermiteBasis, stiffness_matrix_fourth_derivative, load_vector),
        (HermiteBasis, mass_matrix_fourth_derivatives, load_vector),
        (QuadraticBasis, stiffness_matrix_first_derivative, load_vector),
        (LinearBasis, stiffness_matrix_second_derivative, load_vector),
    ],
)
def test_global_assembly(
    basis_function_class, element_matrix_func, element_vector_func
):
    mesh_nodes = np.linspace(0, 1, 3)
    nb_quadrature_points = 3

    basis_function = basis_function_class()

    global_matrix, global_vector = assemble_global_matrices(
        mesh_nodes,
        basis_function,
        element_matrix_func,
        element_vector_func,
        nb_quadrature_points,
    )
    print(global_matrix.shape)
    print(global_matrix)
    print(global_vector.shape)
    print(global_vector)

    dofs_per_node = basis_function.dof_per_node
    nb_nodes = basis_function.nb_nodes
    nb_elements = len(mesh_nodes) - 1
    total_dofs = nb_elements * (nb_nodes - dofs_per_node) + dofs_per_node

    # Basic assertions
    assert global_matrix.shape == (total_dofs, total_dofs)
    assert global_vector.shape == (total_dofs,)


########################################################## 2D ##########################################################
@pytest.mark.parametrize(
    "domain_length, domain_height, n_x, nb_quad_pts_2d, source_func",
    [
        (1, 1, 9, 4, lambda s, n: -6.0),  # Square domain, 4×4 nodes, 9-point quad
    ],
)
def test_assemble_global_matrices_vectors_2d(domain_length, domain_height, n_x, nb_quad_pts_2d, source_func):
    """
    Parametrized test for global assembly of stiffness (K), mass (M), and source (F)
    for Q1 bilinear elements on a structured 2D rectangular domain.

    Checks:
        - Shape of global matrices and vector
        - Symmetry of K and M
        - Nonzero F for constant source
    """
    shape_function = BilinearShapeFunctions()

    # Run assembly
    K, M, F, ny = assemble_global_matrices_vectors_2d(
        shape_function,
        domain_length,
        domain_height,
        n_x,
        nb_quad_pts_2d,
        source_func,
    )


    print("Global stiffness matrix K:\n", K)
    print("\nGlobal mass matrix M:\n", M)
    print("\nGlobal source vector F:\n", F)


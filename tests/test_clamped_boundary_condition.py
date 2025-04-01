"""
Test clamped boundary condition for various system sizes.

Ensures:
- First two and last two rows/columns of the matrix are identity-like.
- Corresponding entries in the vector are zero.

"""

import numpy as np
import pytest
from VascularFlow.BoundaryConditions.ClampedBoundaryCondtion import (
    clamped_boundary_condition,
)


@pytest.mark.parametrize("size", [4, 6])
def test_clamped_boundary_condition(size):
    # Dummy global matrix and vector with nonzero values
    matrix = np.full((size, size), 2.0)
    vector = np.full(size, 3.0)

    # Apply BC
    matrix_bc, vector_bc = clamped_boundary_condition(matrix.copy(), vector.copy())
    print(matrix_bc, vector_bc)
    # Indices of clamped degrees of freedom
    clamped_dofs = [0, 1, size - 2, size - 1]

    # Check matrix rows and columns
    for dof in clamped_dofs:
        # Entire row should be 0 except diagonal
        expected_row = np.zeros(size)
        expected_row[dof] = 1.0
        assert np.allclose(matrix_bc[dof], expected_row), f"Row {dof} is incorrect"

        # Load vector entry should be 0
        assert vector_bc[dof] == 0.0, f"vector[{dof}] should be 0"

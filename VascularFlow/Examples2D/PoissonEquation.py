import numpy as np

from VascularFlow.Numerics.Assembly import assemble_global_matrices_vectors_2d
from VascularFlow.Numerics.FindBoundaryDOFs2D import boundary_dofs_2d
from VascularFlow.Numerics.DirichletBoundaryCondition2D import dirichlet_bc


def poisson_equation_solver(
        shape_function,
        domain_length: float,
        domain_height: float,
        n_x: int,
        bc_positions,
        bc_values,
        source_func: callable = None,
):
    nb_quad_pts_2d = 9

    K, M, F, n_y = assemble_global_matrices_vectors_2d(
        shape_function,
        domain_length,
        domain_height,
        n_x,
        nb_quad_pts_2d,
        source_func,
    )

    dof_positions = boundary_dofs_2d(n_x, n_y, bc_positions)

    lhs_matrix, rhs_vector = dirichlet_bc(K, F, dof_positions, bc_values)

    poisson_equation_solution = np.linalg.solve(lhs_matrix, rhs_vector)

    return poisson_equation_solution



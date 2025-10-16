# -----------------------------------------------------------------------------
# Kirchhoff–Love plate bending (ACM shape function)
# -----------------------------------------------------------------------------
# Purpose
# -------
# Solve a 2D Kirchhoff–Love plate bending problem on a structured rectangular
# grid using an ACM-type (3 DOF per node) finite element.
#
# Unknowns per node (in this code path):
#     - w   : transverse deflection
#     - θx  : rotation about x
#     - θy  : rotation about y
#
# Pipeline
# --------
# 1) Build global matrices/vectors on the reference square with Gaussian quadrature
#    and map to the physical mesh (assemble_global_matrices_vectors_2d_acm).
# 2) Compute the non-dimensional bending stiffness-like scalar D that scales LHS.
# 3) Expand the scalar distributed transverse load q(x,y) from 1 DOF/node to the
#    3-DOF ACM layout by inserting zeros for rotational DOFs.
# 4) Impose Dirichlet boundary conditions (on sets of global DOF indices).
# 5) Solve the sparse linear system with scipy.sparse.linalg.spsolve.
#
# Notes
# -----
# - Dirichlet BCs are enforced via “Galerkin overwrite” (identity rows at DOFs).
# - The distributed transverse load is assumed to act only on w (deflection).
# - All arrays are NumPy; K is expected to be sparse or sparse-compatible.
# -----------------------------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import spsolve

from VascularFlow.Numerics.Assembly import assemble_global_matrices_vectors_2d_acm
from VascularFlow.Numerics.FindBoundaryDOFs2D import boundary_dofs_acm_2d
from VascularFlow.Numerics.DirichletBoundaryCondition2D import dirichlet_bc_acm_2d


def plate_bending_acm_2d(
    shape_function,
    domain_length: float,
    domain_height: float,
    n_x: int,
    n_y: int,
    plate_thickness: float,
    plate_young_modulus: float,
    plate_poisson_ratio: float,
    initial_channel_height: float,
    fluid_density: float,
    fluid_velocity: float,
    bc_positions,
    bc_values,
    distributed_load_channel1: np.array,
    distributed_load_channel2: np.array,
):
    """
    Parameters
    ----------
    shape_function : object
        A shape function instance compatible with the ACM element.
    domain_length : float
        Plate length in the x-direction.
    domain_height : float
        Plate height in the y-direction.
    n_x : int
        Number of nodes along x.
    n_y : int
        Number of nodes along y.
    plate_thickness : float
        Plate thickness t.
    plate_young_modulus : float
        Young’s modulus E of the plate material.
    plate_poisson_ratio : float
        Poisson’s ratio ν of the plate material.
    initial_channel_height : float
        Reference height H0 used in the non-dimensionalization (appears in D).
    fluid_density : float
        Fluid density ρ used in the non-dimensionalization (appears in D).
    fluid_velocity : float
        Fluid velocity U used in the non-dimensionalization (appears in D).
    bc_positions : list[str] or tuple[str, ...]
        Boundary position specifiers understood by `boundary_dofs_acm_2d`
        (e.g., ["bottom", "right", "top", "left"]).
    bc_values : list[float]
        Dirichlet values (one per boundary group). Can also be extended to
        per-DOF values if your BC routine supports it.
    distributed_load_channel1 : np.ndarray, shape (n_x * n_y,)
        Scalar transverse fluid pressure on the top of the plate at each node (applied to w only).
    distributed_load_channel2 : np.ndarray, shape (n_x * n_y,)
        Scalar transverse fluid pressure on the bottom of the plate at each node (applied to w only).

    Returns
    -------
    kirchhoff_Love_plate_solution : np.ndarray, shape (3 * n_x * n_y,)
        Global solution vector ordered by node with 3 DOFs per node: [w, θx, θy].
    lhs_matrix : (N,N) array-like (sparse or dense, depending on assembly)
        The left-hand side matrix after Dirichlet application (and before scaling).
    rhs_vector : (N,-) np.ndarray
        The right-hand side vector after Dirichlet application and load expansion.
    """

    # -------------------------------
    # 1) Compute the non-dimensional scalar multiplier D
    # -------------------------------
    D = (2 * (plate_thickness / 2) ** 3 * plate_young_modulus) / (
        3
        * (1 - plate_poisson_ratio**2)
        * initial_channel_height**3
        * fluid_density
        * fluid_velocity**2
    )
    # Number of Gauss points for element integration (tensor 3×3 rule → 9 points).
    nb_quad_pts_2d = 9
    # -------------------------------
    # 2) Assemble global matrices/vectors (ACM: 3 DOFs per node)
    # -------------------------------
    # K: global matrix; F: global vector (both in ACM DOF ordering).
    K, F = assemble_global_matrices_vectors_2d_acm(
        shape_function,
        domain_length,
        domain_height,
        n_x,
        n_y,
        nb_quad_pts_2d,
    )
    # -------------------------------
    # 3) Boundary conditions (Dirichlet) – determine constrained DOFs and apply
    # -------------------------------
    boundary_dofs = boundary_dofs_acm_2d(n_x, n_y, bc_positions)
    # The BC function returns modified LHS/RHS after enforcing Dirichlet constraints.
    lhs_matrix, rhs_vector = dirichlet_bc_acm_2d(K, F, boundary_dofs, bc_values)
    # distributed_transverse_load is length n_x*n_y and targets only the w DOF.
    expected_size = n_x * n_y
    # Validate distributed loads for both channels
    for name, arr in {
        "distributed_load_channel1": distributed_load_channel1,
        "distributed_load_channel2": distributed_load_channel2,
    }.items():
        if arr.size != expected_size:
            raise ValueError(
                f"{name} must be size n_x * n_y ({expected_size}), "
                f"but got {arr.size}."
            )

    # -------------------------------
    # 4) Build [q, 0, 0, q, 0, 0, ...] vector to match 3 DOFs per node.
    # -------------------------------
    # Function to expand a scalar field (1 DOF/node) to ACM layout (3 DOFs/node)
    def expand_to_acm_layout(load: np.ndarray) -> np.ndarray:
        expanded = np.zeros(load.size * 3, dtype=load.dtype)
        expanded[::3] = load  # assign only the w DOF entries
        return expanded

    # Apply to both channels
    new_distributed_load_channel1 = expand_to_acm_layout(distributed_load_channel1)
    new_distributed_load_channel2 = expand_to_acm_layout(distributed_load_channel2)
    new_distributed_load = new_distributed_load_channel2 - new_distributed_load_channel1

    # -------------------------------
    # 5) Form system and solve
    # -------------------------------
    # Scale the LHS with D and multiply the RHS by the expanded load weights.
    a = D * lhs_matrix
    l = new_distributed_load * rhs_vector
    # Solve the system; spsolve accepts sparse or dense as long as it is array-like.
    kirchhoff_Love_plate_solution = spsolve(a, l)

    return kirchhoff_Love_plate_solution, lhs_matrix, rhs_vector

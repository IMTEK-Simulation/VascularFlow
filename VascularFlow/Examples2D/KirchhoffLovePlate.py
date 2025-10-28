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

    # -------------------------------------------------------------------------
    # 1) Non-dimensional bending scale D (kept scalar here).
    #    This comes from classic plate bending: Eh^3 / [12(1-ν^2)] scaled by (ρ U^2 H0^3)^-1
    #    so that final linear system is dimensionless. If you switch to fully
    #    dimensional K (with constitutive coupling), set D=1 and put material in K.
    # -------------------------------------------------------------------------
    D = (plate_thickness ** 3 * plate_young_modulus) / (
            12
            * (1 - plate_poisson_ratio ** 2)
            * initial_channel_height ** 3
            * fluid_density
            * fluid_velocity ** 2
    )

    # Use 3x3 tensor-product Gauss rule (9 points) for the element integrals.
    nb_quad_pts_2d = 9

    # -------------------------------------------------------------------------
    # 2) Assemble global stiffness K and nodal weight vector (∫ N_i dA) in ACM layout.
    #    K is built from mapped second derivatives; weights_vec contains the
    #    consistent "mass-like" nodal areas for w-DOF expansion later.
    # -------------------------------------------------------------------------
    K, weights_vec = assemble_global_matrices_vectors_2d_acm(
        shape_function,
        domain_length,
        domain_height,
        n_x,
        n_y,
        nb_quad_pts_2d,
    )

    # -------------------------------------------------------------------------
    # 3) Boundary DOFs: which global indices are constrained.
    #    We do NOT apply BCs yet; first we build the physical RHS.
    # -------------------------------------------------------------------------
    boundary_dofs = boundary_dofs_acm_2d(n_x, n_y, bc_positions)

    # -------------------------------------------------------------------------
    # 4) Build the distributed transverse load on ACM layout.
    #    - Validate sizes (always helpful to catch indexing bugs).
    #    - Expand q from (w-only) to [w, θx, θy] pattern: [q, 0, 0, q, 0, 0, ...].
    #    - Form L by nodal lumping: L_i = q_i * ∫ N_i dA (already in ACM ordering).
    #    IMPORTANT: Build L BEFORE applying BCs so total load is mesh-independent.
    # -------------------------------------------------------------------------
    expected_size = n_x * n_y
    for name, arr in {
        "distributed_load_channel1": distributed_load_channel1,
        "distributed_load_channel2": distributed_load_channel2,
    }.items():
        if arr.size != expected_size:
            raise ValueError(
                f"{name} must be size n_x * n_y ({expected_size}), got {arr.size}."
            )

    def expand_to_acm_layout(load: np.ndarray) -> np.ndarray:
        """Map a scalar field (per node) to ACM DOF ordering: fill w, zero rotations."""
        out = np.zeros(load.size * 3, dtype=load.dtype)
        out[::3] = load  # w-DOF entries
        return out

    # Effective pressure (channel2 acting opposite to channel1) at nodes:
    q = distributed_load_channel2 - distributed_load_channel1  # (n_x*n_y,)
    q_acm = expand_to_acm_layout(q)  # (3*n_x*n_y,)

    # Lump to nodal RHS using previously assembled weights (same ACM ordering):
    L = q_acm * weights_vec

    # -------------------------------------------------------------------------
    # 5) Apply Dirichlet boundary conditions to BOTH K and L.
    #    Note: current routine zeros rows only (non-symmetric BC application).
    #    If you need symmetry, also zero columns and adjust RHS accordingly.
    # -------------------------------------------------------------------------
    K_bc, L_bc = dirichlet_bc_acm_2d(K.tolil(), L.copy(), boundary_dofs, bc_values)

    # -------------------------------------------------------------------------
    # 6) Solve the linear system. Multiply K by D, keep RHS as built.
    #    Use CSR for spsolve; for iterative solvers consider preconditioning
    #    because biharmonic problems are ill-conditioned as the mesh refines.
    # -------------------------------------------------------------------------
    A = D * K_bc  # final LHS
    kirchhoff_Love_plate_solution = spsolve(A.tocsr(), L_bc)

    return kirchhoff_Love_plate_solution, K_bc, L_bc

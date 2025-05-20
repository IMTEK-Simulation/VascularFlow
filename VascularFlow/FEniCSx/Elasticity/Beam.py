"""
Steady-state Euler-Bernoulli beam solver using FEniCSx.

This module provides a function to solve the dimensionless steady-state Euler-Bernoulli beam equation over
a solid domain using FEniCSx. The formulation accounts for both:
    -bending (linear term)
    -extensional stiffness (non-linear term)
and supports externally applied distributed loads (e.g., in a dual micro-channel with a solid boundary in between).

The problem is typically used in fluid-structure interaction (FSI) or multi-physics simulations where a
flexible solid structure is deformed by loads induced from a surrounding environment.

States:
    - steady (β ∂⁴w/∂x⁴ - γ(∂w/∂x)² ∂2w/∂x2 = p1 - p-2; in solid domain)
    - w: Displacement of the solid boundary.
    - β: Dimensionless bending stiffness of the beam.
    - γ: Dimensionless extensional stiffness of the beam.
    - p1: Distributed external loads on the beam applied from the lower channel.
    - p2: Distributed external loads on the beam applied from the upper channel.

Boundary conditions:
    - zero displacement and rotation at beam ends
"""

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import functionspace
import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import nls, log


def euler_bernoulli_steady_fenicsx(
    solid_domain: mesh.Mesh,
    solid_domain_x_max_coordinate: int,
    dimensionless_bending_stiffness: float,
    dimensionless_extensional_stiffness: float,
    distributed_load_channel2: np.ndarray,
    distributed_load_channel1: np.ndarray,
    linera: bool,
):
    """
    Solves the steady-state Euler-Bernoulli beam equation on a solid domain using FEniCSx.

    Parameters
    ----------
    solid_domain : mesh.Mesh
        The finite element mesh representing the solid domain where the beam equation is solved.
    solid_domain_x_max_coordinate : int
        The x-coordinate of the right (maximum x) boundary of the solid domain.
    dimensionless_bending_stiffness : float
        Dimensionless parameter representing the beam's bending stiffness.

    dimensionless_extensional_stiffness : float
        Dimensionless parameter representing the beam's axial (extensional) stiffness.

    distributed_load_channel2 : np.ndarray
        Nodal values of distributed external loads applied from channel 2(lower channel),
        representing fluid pressure.

    distributed_load_channel1 : np.ndarray
        Nodal values of distributed external loads applied from channel 1(upper channel),
        representing fluid pressure which act in opposition with channel 2.

    linera : bool
        Flag indicating whether the solver should use a linear or nonlinear formulation.
        If True, assumes a linear model. If False, includes nonlinear effects.
    Returns
    -------
    displacement_function : np.ndarray
        The computed displacement field of the beam, defined over the solid domain.

    Notes
    -----
    This function solves a dimensionless form of the Euler-Bernoulli equation,
    which assumes small deformations and negligible shear deformation.
    The results can be used to couple solid deformation back into fluid models in
    fluid–structure interaction (FSI) simulations, serving as boundary conditions on the top wall of the fluid domain
    to compute the corresponding mesh deformation.
    """
    # Defining the constant values over the solid domain
    alpha = fem.Constant(
        solid_domain, default_scalar_type(dimensionless_bending_stiffness)
    )
    beta = fem.Constant(
        solid_domain, default_scalar_type(dimensionless_extensional_stiffness)
    )

    # Defining the finite element function space using Hermite basis function
    beam_element = basix.ufl.element(
        basix.ElementFamily.Hermite, basix.CellType.interval, 3
    )
    V = functionspace(solid_domain, beam_element)

    # Defining the trial and test function
    uh = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Defining boundary conditions (zero displacement and rotation at beam ends)
    boundary_condition_function = fem.Function(V)

    start_point = locate_entities_boundary(
        solid_domain, 0, lambda x: np.isclose(x[0], 0)
    )
    end_point = locate_entities_boundary(
        solid_domain, 0, lambda x: np.isclose(x[0], solid_domain_x_max_coordinate)
    )
    start_dof = fem.locate_dofs_topological(V, 0, start_point)
    end_dof = fem.locate_dofs_topological(V, 0, end_point)

    fixed_disp_start_point = fem.dirichletbc(
        boundary_condition_function, np.array([start_dof[0]])
    )
    fixed_rot_start_point = fem.dirichletbc(
        boundary_condition_function, np.array([start_dof[1]])
    )
    fixed_disp_end_point = fem.dirichletbc(
        boundary_condition_function, np.array([end_dof[0]])
    )
    fixed_rot_end_point = fem.dirichletbc(
        boundary_condition_function, np.array([end_dof[1]])
    )

    BCs = [
        fixed_disp_start_point,
        fixed_rot_start_point,
        fixed_disp_end_point,
        fixed_rot_end_point,
    ]

    # Define and assign distributed load functions for channel 2 (top) and channel 1 (bottom)
    q_channel2 = fem.Function(V)
    q_channel2.x.array[::2] = distributed_load_channel2

    q_channel1 = fem.Function(V)
    q_channel1.x.array[::2] = distributed_load_channel1

    if linera is True:
        # Forming and solving the linear system
        a = alpha * ufl.dot(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
        L = q_channel2 * v * ufl.dx - q_channel1 * v * ufl.dx
        problem = LinearProblem(
            a, L, bcs=BCs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
    else:
        # Forming and solving the non-linear system
        def non_linear_term(u):
            return ufl.grad(u) ** 2

        F = alpha * ufl.dot(ufl.div(ufl.grad(uh)), ufl.div(ufl.grad(v))) * ufl.dx
        F -= beta * non_linear_term(uh) * ufl.dot(ufl.div(ufl.grad(uh)), v) * ufl.dx
        F -= q_channel2 * v * ufl.dx
        F += q_channel1 * v * ufl.dx
        problem = NonlinearProblem(F, uh, bcs=BCs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.report = False

        log.set_log_level(log.LogLevel.WARNING)
        n, converged = solver.solve(uh)
        assert converged
        print(f"Nonlinear Euler–Bernoulli beam solver converged in: {n:d} iterations")

    # Extract displacement and rotation values from the mixed solution `uh` by splitting the array:
    # even-indexed entries correspond to displacement, and odd-indexed entries correspond to rotation
    displacement = np.empty(0)
    rotation = np.empty(0)
    for i, x in enumerate(uh.x.array):
        if i % 2 != 0:
            rotation = np.append(rotation, x)
        else:
            displacement = np.append(displacement, x)

    return displacement

"""
Harmonic extension (HE) and bi-Harmonic extension (BE) implementation to extend the displacement of the FSI interface
    into the fluid domain.
Given the interface displacement w, the ALE displacement w_a (mesh displacement) is computed by solving:
-laplace’s equation (for harmonic extension implementation)
-bi-harmonic equation (for bi-harmonic extension implementation)
in the initial configuration of the fluid domain for every displacement component.

States:
- steady (Δw_a= 0 in fluid domain)
- steady (Δ²w_a= 0 in fluid domain)

Boundary conditions:
1) Harmonic extension implementation
    - Dirichlet boundary condition on the FSI interface (beam displacement on the fluid top wall)
    - zero displacement at the rest of the boundary
2) Bi-harmonic extension implementation
    - Dirichlet boundary condition on the FSI interface (beam displacement on the fluid top wall)
    - zero displacement at the rest of the boundary
    - zero Laplacian of the mesh displacement on the fluid walls(Δw_a).
"""

from dolfinx import mesh, fem, default_scalar_type
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import functionspace


def mesh_deformation(
    interface_displacement: np.ndarray,
    fluid_domain_x_max_coordinate: int,
    fluid_domain: mesh.Mesh,
    harmonic_extension: bool,
):
    """
    Extend displacement of the FSI interface into the fluid domain by means of:
        -harmonic extension (HE) method
        -bi-harmonic extension (BE) method

    This function solves a Laplace or bi-harmonic problem to smoothly propagate the prescribed displacement from the
    elastic interface (the top wall) into the interior of the fluid mesh. It is commonly used in fluid-structure
    interaction (FSI) problems to update the fluid mesh in response to structural motion.

    Parameters
    ----------
    interface_displacement : np.ndarray
        Nodal displacement values prescribed along the FSI interface (top wall).
        This array should match the number of degrees of freedom (DOFs) along the moving boundary.
    fluid_domain : mesh.Mesh
        The fluid mesh domain to be deformed. This is a 2D mesh representing the fluid region in the FSI setup.
    fluid_domain_x_max_coordinate: int
        The x-coordinate of the right (maximum x) boundary of the fluid domain.
    harmonic_extension : bool
        Switch between Harmonic extension implementation and bi-harmonic extension.
    Returns
    ----------
    deformed_domain : mesh.Mesh
        The input fluid mesh, updated in place with the harmonic extension of the beam displacement.
        The mesh geometry (coordinates) is modified by applying the computed harmonic field.
    Notes
    -------
    - This function assumes a scalar harmonic extension along a single coordinate direction
    (e.g., vertical movement in y-direction).
    - This method ensures smooth transitions in mesh deformation.
    """

    # Defining the finite element function space
    displacement_function_space = functionspace(fluid_domain, ("Lagrange", 1))

    # Defining dirichlet boundary conditions
    def walls_rest(x):
        return (
            np.isclose(x[0], 0)
            | np.isclose(x[0], fluid_domain_x_max_coordinate)
            | np.isclose(x[1], 0)
        )

    tdim = fluid_domain.topology.dim
    fdim = tdim - 1
    fluid_domain.topology.create_connectivity(
        fluid_domain.topology.dim - 1, fluid_domain.topology.dim
    )
    facets_rest = mesh.locate_entities_boundary(fluid_domain, fdim, walls_rest)

    all_boundary_facets = mesh.exterior_facet_indices(fluid_domain.topology)
    facet_top = np.setdiff1d(all_boundary_facets, facets_rest)

    boundary_dofs_rest = fem.locate_dofs_topological(
        displacement_function_space, fdim, facets_rest
    )
    boundary_dofs_top = fem.locate_dofs_topological(
        displacement_function_space, fdim, facet_top
    )

    bc_rest = fem.dirichletbc(
        default_scalar_type(0), boundary_dofs_rest, displacement_function_space
    )

    top_wall_bc_value = fem.Function(displacement_function_space)
    # Implementing the transfer of fluid-structure interaction (FSI) interface displacements
    # into the degrees of freedom (DoFs) of the top wall of the fluid domain.
    for dof, value in zip(boundary_dofs_top, interface_displacement):
        top_wall_bc_value.x.array[dof] = value
    bc_top = fem.dirichletbc(top_wall_bc_value, boundary_dofs_top)
    bc = [bc_top, bc_rest]

    if harmonic_extension is True:
        # Defining the variational problem
        u = ufl.TrialFunction(displacement_function_space)
        v = ufl.TestFunction(displacement_function_space)

        lhs = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        rhs = fem.Constant(fluid_domain, default_scalar_type(0)) * v * ufl.dx

        # solving the Laplace’s equation
        problem = LinearProblem(
            lhs, rhs, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        w_a = problem.solve()
    else:
        # Defining the penalty parameter 'alpha' and the cell diameter 'h' for
        # Weak imposition of dirichlet conditions for the bi-harmonic problem using Nitsche’s method
        alpha = default_scalar_type(8)
        h = ufl.CellDiameter(fluid_domain)
        n = ufl.FacetNormal(fluid_domain)
        h_avg = (h("+") + h("-")) / 2.0

        # Define variational problem
        u = ufl.TrialFunction(displacement_function_space)
        v = ufl.TestFunction(displacement_function_space)
        f = fem.Constant(fluid_domain, default_scalar_type(0))

        a = (
            ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
            - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(v), n))
            * ufl.dS
            - ufl.inner(ufl.jump(ufl.grad(u), n), ufl.avg(ufl.div(ufl.grad(v))))
            * ufl.dS
            + alpha
            / h_avg
            * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n))
            * ufl.dS
        )
        L = ufl.inner(f, v) * ufl.dx

        problem = LinearProblem(
            a, L, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        w_a = problem.solve()

    # Applying the vertical movement in y-direction
    x = fluid_domain.geometry.x
    x[:, 1] += w_a.x.array
    return fluid_domain


def mesh_deformation_3d(
    interface_displacement: np.ndarray,
    fluid_domain: mesh.Mesh,
    fluid_domain_x_max_coordinate: float,
    fluid_domain_y_max_coordinate: float,
):
    """
    Build a hexahedral unit-cube mesh, scale it to a rectangular prism, and
    solve a harmonic extension problem (Laplace’s equation) to vertically
    displace the mesh using a user-specified Dirichlet profile on the top face.

    The scalar field u is then extended into the volume by solving
        -Δu = 0 in Ω,  u|∂Ω as above,
        and the mesh geometry is updated by adding u to the z-coordinate.

    Parameters
    ----------
    interface_displacement : np.ndarray
        Flattened array of length (n_y + 1) * (n_x + 1) containing the
        prescribed values on the top face DOFs of a CG1 space. The values
        must be ordered as:
            for y in [Ly, Ly - Δy, ..., 0]:
                for x in [Lx, Lx - Δx, ..., 0]:
                    append u(x, y)
        where Δx = Lx / n_x and Δy = Ly / n_y.
        (I.e., first row is y = Ly with x descending; last row is y = 0.)
    fluid_domain : dolfinx.mesh.Mesh
        A 3D finite element mesh defining the geometry of the fluid domain.
    fluid_domain_x_max_coordinate : float
        Physical length in the x-direction (Lx).
    fluid_domain_y_max_coordinate : float
        Physical length in the y-direction (Ly).

    Returns
    -------
    dolfinx.mesh.Mesh
        The deformed mesh. Only the z-coordinates are modified in-place:
        X[:, 2] ← X[:, 2] + u.
    """

    tdim = fluid_domain.topology.dim
    fdim = tdim - 1
    fluid_domain.topology.create_connectivity(tdim, tdim)

    # Scalar CG1 space for the harmonic extension
    displacement_function_space = functionspace(fluid_domain, ("Lagrange", 1))

    # Identify all boundary facets except the top face (z = Lz) → these are "rest" (fixed)
    def walls_rest(x):
        return (
            np.isclose(x[0], 0)
            | np.isclose(x[0], fluid_domain_x_max_coordinate)
            | np.isclose(x[1], 0)
            | np.isclose(x[1], fluid_domain_y_max_coordinate)
            | np.isclose(x[2], 0)
        )

    facets_rest = mesh.locate_entities_boundary(fluid_domain, fdim, walls_rest)
    all_boundary_facets = mesh.exterior_facet_indices(fluid_domain.topology)
    facet_top = np.setdiff1d(all_boundary_facets, facets_rest)

    # Dirichlet DOFs on the non-top boundary (fixed to zero)
    boundary_dofs_rest = fem.locate_dofs_topological(
        displacement_function_space, fdim, facets_rest
    )

    boundary_dofs_top = fem.locate_dofs_topological(
        displacement_function_space, fdim, facet_top
    )
    dof_coordinates = displacement_function_space.tabulate_dof_coordinates()
    top_wall_dofs_coords = dof_coordinates[boundary_dofs_top]

    ys = np.round(top_wall_dofs_coords[:, 1], 12)
    y_levels = np.unique(ys)[::-1]
    sorted_indices = []
    for y in y_levels:
        idx = np.where(ys == y)[0]
        idx_sorted = idx[np.argsort(-top_wall_dofs_coords[idx, 0])]
        sorted_indices.extend(idx_sorted)

    sorted_indices = np.array(sorted_indices)
    sorted_boundary_dofs_top = boundary_dofs_top[sorted_indices]
    # Build Dirichlet data: zeros everywhere, custom values on top DOFs
    bc_rest = fem.dirichletbc(
        default_scalar_type(0), boundary_dofs_rest, displacement_function_space
    )
    top_wall_bc_value = fem.Function(displacement_function_space)
    # Assign user-specified top values in the requested order
    for dof, value in zip(sorted_boundary_dofs_top, interface_displacement):
        top_wall_bc_value.x.array[dof] = value
    bc_top = fem.dirichletbc(top_wall_bc_value, sorted_boundary_dofs_top)
    bc = [bc_top, bc_rest]

    # Variational problem: -Δu = 0 with Dirichlet BCs
    u = ufl.TrialFunction(displacement_function_space)
    v = ufl.TestFunction(displacement_function_space)
    lhs = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = fem.Constant(fluid_domain, default_scalar_type(0)) * v * ufl.dx

    # Solve Laplace problem
    problem = LinearProblem(
        lhs, rhs, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    w_a = problem.solve()

    # Update mesh geometry by adding u to z-coordinate (vertical displacement)
    x = fluid_domain.geometry.x
    x[:, 2] += w_a.x.array

    return fluid_domain

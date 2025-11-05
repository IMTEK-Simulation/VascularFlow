"""
Steady-state Navier–Stokes equations solver using FEniCSx.

This module provides a function to solve the dimensionless steady-state Navier–Stokes equations over
a fluid domain using FEniCSx. The implementation supports both:
    -Navier–Stokes equations: Includes the nonlinear convective term, capturing full fluid dynamics behavior.
    -Stokes equations: A simplified version that omits the nonlinear convective term,
        suitable for low Reynolds number (creeping flow) regimes

The problem is typically used in fluid-structure interaction (FSI) or multi-physics simulations where
a viscous fluid flows through a micro-channel and interacts with deformable structural boundaries

States:
    - steady Navier–Stokes equations:
        u·∇u = −∇p + Re−1 ∇²u
        ∇·u = 0
    - steady Stokes equations:
        ∇p - ∇²u = 0
        ∇·u = 0
    - u: Velocity field of the fluid (vector quantity)
    - p: Pressure field of the fluid (scalar quantity)
    - Re: Reynolds number, a dimensionless quantity representing the ratio of inertial to viscous forces

Boundary conditions:
    - A constant pressure is applied at the inlet (left boundary).
    - Zero pressure is applied at the outlet (right boundary).
    - No-slip (zero velocity) boundary conditions are applied at the top and bottom walls.
    - The flow is driven by the pressure difference.
"""

from mpi4py import MPI
import dolfinx
import basix.ufl
import ufl
from petsc4py import PETSc
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import log
import scipy.sparse


def pressure_2d_pressure_inlet(
    fluid_domain: dolfinx.mesh.Mesh,
    inlet_coordinate: float,
    outlet_coordinate: float,
    reynolds_number: float,
    inlet_pressure: float,
    navier_stokes: bool,
):
    """
    Solve the 2D incompressible Navier-Stokes equations and Stokes equations for a fluid domain with
    prescribed pressure inlet and outlet, and no-slip walls using FEniCSx.

    Parameters
    ----------
    fluid_domain : dolfinx.mesh.Mesh
        A 2D finite element mesh defining the geometry of the fluid domain.
    inlet_coordinate : float
        The x-coordinate identifying the inlet boundary of the fluid domain where the inlet pressure condition is applied
    outlet_coordinate : float
        The x-coordinate identifying the outlet boundary of the fluid domain where the outlet pressure condition is applied.
    reynolds_number : float
        The Reynolds number used in the Navier-Stokes equations.
    inlet_pressure : float
        The scalar pressure value to be imposed as a Dirichlet boundary condition at the fluid domain’s inlet
    navier_stokes : bool
        If True (default), solves the full Navier–Stokes equations including the nonlinear
        convective term. If False, solves the simplified Stokes equations without the convective term.

    Returns
    -------
    dolfinx.fem.Function
        The solution function `wh`, which is a mixed finite element function
        containing both the velocity (`uh`) and pressure (`ph`) components.

    Notes
    -----
    - The velocity is discretized using a second-order Lagrange element.
    - The pressure is discretized using a first-order Lagrange element.
    - A Newton solver is used to solve the nonlinear system.
    - The function prints the number of iterations and convergence status of the nonlinear system.
    """

    # Defining finite elements and mixed function space
    el_u = basix.ufl.element(
        "Lagrange", fluid_domain.topology.cell_name(), 2, shape=(2,)
    )
    el_p = basix.ufl.element("Lagrange", fluid_domain.topology.cell_name(), 1)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])

    # Defining test and trial functions in mixed spaces
    W = dolfinx.fem.functionspace(fluid_domain, el_mixed)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    #  Definition of integration term over exterior facets in fluid mesh and the outwards pointing facet normal
    ds = ufl.Measure("ds", domain=fluid_domain)
    n = ufl.FacetNormal(fluid_domain)

    # Defining the velocity subspace (used in boundary condition section)
    W0 = W.sub(0)
    V, _ = W0.collapse()

    # Defining the solution function
    wh = dolfinx.fem.Function(W)
    uh, ph = ufl.split(wh)

    # Locating a subset of entities on a boundary
    fluid_domain.topology.create_connectivity(
        fluid_domain.topology.dim - 1, fluid_domain.topology.dim
    )

    def inlet_marker(x):
        return np.isclose(x[0], inlet_coordinate)

    inlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, inlet_marker
    )

    def outlet_marker(x):
        return np.isclose(x[0], outlet_coordinate)

    outlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, outlet_marker
    )

    def bottom_marker(x):
        return np.isclose(x[1], 0)

    bottom_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, bottom_marker
    )

    all_boundary_facets = dolfinx.mesh.exterior_facet_indices(fluid_domain.topology)
    top_facet = np.setdiff1d(
        all_boundary_facets,
        np.unique(np.concatenate((inlet_facet, outlet_facet, bottom_facet))),
    )

    # Dirichlet conditions in mixed spaces
    p_inlet = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(inlet_pressure)
    )
    dofs_inlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, inlet_facet
    )
    bc_inlet = dolfinx.fem.dirichletbc(p_inlet, dofs_inlet, W.sub(1))

    p_outlet = dolfinx.fem.Constant(fluid_domain, dolfinx.default_scalar_type(0))
    dofs_outlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, outlet_facet
    )
    bc_outlet = dolfinx.fem.dirichletbc(p_outlet, dofs_outlet, W.sub(1))

    u_wall = dolfinx.fem.Function(V)
    u_wall.x.array[:] = 0
    dofs_wall_bottom = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, bottom_facet
    )
    bc_wall_bottom = dolfinx.fem.dirichletbc(u_wall, dofs_wall_bottom, W0)

    dofs_wall_top = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, top_facet
    )
    bc_wall_top = dolfinx.fem.dirichletbc(u_wall, dofs_wall_top, W0)

    bcs = [bc_inlet, bc_outlet, bc_wall_bottom, bc_wall_top]

    if navier_stokes is True:
        # Defining the Reynolds number over the fluid domain
        Re = dolfinx.fem.Constant(
            fluid_domain, dolfinx.default_scalar_type(reynolds_number)
        )

        # Variational form
        F = ufl.inner(ufl.grad(uh) * uh, v) * ufl.dx  # Convective term
        F += (
            (1 / Re) * ufl.inner(ufl.grad(uh), ufl.grad(v))
        ) * ufl.dx  # Diffusion term
        F -= ufl.inner(ph, ufl.div(v)) * ufl.dx  # Pressure gradient
        F += ufl.dot(ph * n, v) * ds  # Weak imposition of Dirichlet conditions
        F += ufl.inner(ufl.div(uh), q) * ufl.dx  # Continuity equation

        # Create the nonlinear problem
        problem = NonlinearProblem(F, wh, bcs)
        # Create the Newton solver
        newton_solver = NewtonSolver(MPI.COMM_WORLD, problem)
        # Set the Newton solver parameters
        newton_solver.convergence_criterion = "incremental"
        newton_solver.rtol = 1e-6
        # newton_solver.max_it = 100
        newton_solver.report = False
        # Modify the linear solver in each Newton iteration
        ksp = newton_solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()
        # Solve the nonlinear problem
        log.set_log_level(log.LogLevel.WARNING)
        n, converged = newton_solver.solve(wh)
        print(
            f"Newton solver for The Navier-Stokes equations completed in {n} iterations. Converged: {converged}"
        )

    else:
        # Defining the source term and reynolds number over the fluid domain
        f = dolfinx.fem.Constant(fluid_domain, dolfinx.default_scalar_type((0, 0)))

        # Variational form
        F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Diffusion term
        F -= ufl.dot(ufl.grad(u) * n, v) * ds
        F -= ufl.inner(p, ufl.div(v)) * ufl.dx  # Pressure gradient
        F += ufl.dot(p * n, v) * ds
        F += ufl.inner(ufl.div(u), q) * ufl.dx  # Continuity equation
        F -= ufl.inner(f, v) * ufl.dx  # Source term
        a, L = ufl.system(F)

        # Compile the bi-linear and linear forms for assembly
        a_compiled = dolfinx.fem.form(a)
        L_compiled = dolfinx.fem.form(L)
        # Create the global system matrix A and right-hand side vector b
        A = dolfinx.fem.create_matrix(a_compiled)
        b = dolfinx.fem.create_vector(L_compiled)
        # Convert the DOLFINx PETSc matrix to a SciPy sparse matrix (for direct solve)
        A_scipy = A.to_scipy()
        # Assemble the global matrix A with boundary conditions
        dolfinx.fem.assemble_matrix(A, a_compiled, bcs)
        # Assemble the global right-hand side vector b
        dolfinx.fem.assemble_vector(b.array, L_compiled)
        # Apply boundary condition contributions (lifting) to the RHS vector
        dolfinx.fem.apply_lifting(b.array, [a_compiled], [bcs])
        # Finalize assembly of b (synchronize parallel data)
        b.scatter_reverse(dolfinx.la.InsertMode.add)
        # Apply Dirichlet boundary values directly to the RHS
        [bc.set(b.array) for bc in bcs]
        # Factorize the system matrix using a sparse LU solver (SciPy)
        A_inv = scipy.sparse.linalg.splu(A_scipy)
        # Create the solution function and solve the linear system
        wh = dolfinx.fem.Function(W)
        wh.x.array[:] = A_inv.solve(b.array)

    # Pressure values on the top wall
    pressure_field = wh.sub(1).collapse()
    dof_coordinates = pressure_field.function_space.tabulate_dof_coordinates()

    top_wall_dofs = dolfinx.fem.locate_dofs_topological(
        pressure_field.function_space, fluid_domain.topology.dim - 1, top_facet
    )

    top_wall_coords = dof_coordinates[top_wall_dofs]
    interface_pressure_values = pressure_field.x.array[top_wall_dofs]
    sorted_indices = np.argsort(top_wall_coords[:, 0])
    interface_pressure = interface_pressure_values[sorted_indices]

    return wh, interface_pressure


def pressure_3d_pressure_inlet(
    fluid_domain: dolfinx.mesh.Mesh,
    fluid_domain_x_inlet_coordinate: float,
    fluid_domain_x_outlet_coordinate: float,
    fluid_domain_y_max_coordinate: float,
    n_x: int,
    n_y: int,
    reynolds_number: float,
    inlet_pressure: float,
):
    """
    Solve steady incompressible Navier–Stokes in a 3D rectangular duct with
    a pressure inlet/outlet and no-slip walls, then return the mixed solution
    and a 1D vector of pressure values on the top face (z = Lz) sorted row-wise.

    Boundary conditions
    -------------------
    - Inlet  (x = Lx): p = inlet_pressure  (Dirichlet on pressure)
    - Outlet (x = 0) : p = 0               (Dirichlet on pressure)
    - Walls  (z = 0, z = Lz, y = 0, y = Ly): u = 0 (no-slip velocity)

    Discretization
    --------------
    - Mesh: hexahedral unit cube scaled to [0,Lx]×[0,Ly]×[0,Lz], with (n_x, n_y, n_z) cells.
    - Elements: Taylor–Hood (Q2 for velocity, Q1 for pressure) via Basix.
    - Nonlinear solve: Newton with LU (MUMPS).

    Returned top-face pressure ordering
    -----------------------------------
    `interface_pressure_values_sorted` is concatenated row-by-row in this order:
      for y in [Ly, Ly - Δy, ..., 0]:    # descending y
        for x in [Lx, Lx - Δx, ..., 0]:  # descending x
          append p(x, y, z=Lz)
    where Δx = Lx/n_x and Δy = Ly/n_y. Each row is first sorted by x (descending).

    Parameters
    ----------
    fluid_domain : dolfinx.mesh.Mesh
        A 3D finite element mesh defining the geometry of the fluid domain.
    fluid_domain_x_inlet_coordinate: float
        The x-coordinate specifying the inlet position of the  fluid domain.
    fluid_domain_x_outlet_coordinate: float
        The x-coordinate specifying the outlet position of the fluid domain.
    fluid_domain_y_max_coordinate: float
        The y-coordinate specifying the maximum width of the fluid domain.
    n_x, n_y : int
        Number of hexahedral cells in x, y.
    reynolds_number : float
        Reynolds number used in the weak form (dimensionless).
    inlet_pressure : float
        Imposed pressure value at the inlet boundary (x = Lx).

    Returns
    -------
    wh : dolfinx.fem.Function
        Mixed solution in W = [Q2]^3 × Q1; `wh.sub(0)` is velocity, `wh.sub(1)` is pressure.
    interface_pressure_values_sorted : np.ndarray
        1D array of length (n_y+1) * (n_x+1) with top-face (z=Lz) pressure
        values sorted as described above.
    """

    # --- Taylor–Hood mixed element: Q2 velocity (vector), Q1 pressure (scalar) ---
    el_u = basix.ufl.element(
        "Lagrange",
        fluid_domain.topology.cell_name(),
        2,
        shape=(fluid_domain.geometry.dim,),
    )
    el_p = basix.ufl.element("Lagrange", fluid_domain.topology.cell_name(), 1)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])

    W = dolfinx.fem.functionspace(fluid_domain, el_mixed)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    ds = ufl.Measure("ds", domain=fluid_domain)
    n = ufl.FacetNormal(fluid_domain)

    # Velocity subspace handle (W.sub(0)); V is the collapsed scalar space for applying BCs
    W0 = W.sub(0)
    V, _ = W0.collapse()

    # Mixed unknown function (for nonlinear solve)
    wh = dolfinx.fem.Function(W)
    uh, ph = ufl.split(wh)

    # --- Boundary marking: inlet (x=Lx), outlet (x=0), bottom (z=0), top (rest of exterior minus inlet/outlet/bottom)
    fluid_domain.topology.create_connectivity(
        fluid_domain.topology.dim - 1, fluid_domain.topology.dim
    )

    def inlet_marker(x):
        return np.isclose(x[0], fluid_domain_x_inlet_coordinate)

    inlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, inlet_marker
    )

    def outlet_marker(x):
        return np.isclose(x[0], fluid_domain_x_outlet_coordinate)

    outlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, outlet_marker
    )

    def walls_rest(x):
        return (
                np.isclose(x[1], 0)
                | np.isclose(x[1], fluid_domain_y_max_coordinate)
                | np.isclose(x[2], 0)
        )

    walls_rest_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, walls_rest
    )

    all_boundary_facets = dolfinx.mesh.exterior_facet_indices(fluid_domain.topology)
    # walls facets = everything else on the exterior (used for no-slip on velocity)
    top_facet = np.setdiff1d(
        all_boundary_facets,
        np.unique(np.concatenate((inlet_facet, outlet_facet, walls_rest_facet))),
    )
    # --- Pressure Dirichlet BCs: inlet/outlet pressures on W.sub(1) ---
    p_inlet = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(inlet_pressure)
    )
    dofs_inlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, inlet_facet
    )
    bc_inlet = dolfinx.fem.dirichletbc(p_inlet, dofs_inlet, W.sub(1))

    p_outlet = dolfinx.fem.Constant(fluid_domain, dolfinx.default_scalar_type(0))
    dofs_outlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, outlet_facet
    )
    bc_outlet = dolfinx.fem.dirichletbc(p_outlet, dofs_outlet, W.sub(1))
    # --- Velocity no-slip on bottom and top walls: u = 0 on W.sub(0) ---
    u_wall = dolfinx.fem.Function(V)
    u_wall.x.array[:] = 0
    dofs_walls_rest = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, walls_rest_facet
    )
    bc_walls_rest = dolfinx.fem.dirichletbc(u_wall, dofs_walls_rest, W0)

    dofs_wall_top = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, top_facet
    )
    bc_wall_top = dolfinx.fem.dirichletbc(u_wall, dofs_wall_top, W0)
    bcs = [bc_inlet, bc_outlet, bc_walls_rest, bc_wall_top]
    # --- Reynolds number parameter ---
    Re = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(reynolds_number)
    )

    # --- Weak form of steady NS (do-nothing form for pressure BCs via +∫ p n·v ds) ---
    F = ufl.inner(ufl.grad(uh) * uh, v) * ufl.dx  # Convective term
    F += ((1 / Re) * ufl.inner(ufl.grad(uh), ufl.grad(v))) * ufl.dx  # Diffusion term
    F -= ufl.inner(ph, ufl.div(v)) * ufl.dx  # Pressure gradient
    F += ufl.dot(ph * n, v) * ds  # Weak imposition of Dirichlet conditions
    F += ufl.inner(ufl.div(uh), q) * ufl.dx  # Continuity equation

    # Create the nonlinear problem
    problem = NonlinearProblem(F, wh, bcs)
    # Create the Newton solver
    newton_solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # Set the Newton solver parameters
    newton_solver.convergence_criterion = "incremental"
    newton_solver.rtol = 1e-6
    # newton_solver.max_it = 100
    newton_solver.report = False
    # Modify the linear solver in each Newton iteration
    ksp = newton_solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    # Solve the nonlinear problem
    log.set_log_level(log.LogLevel.WARNING)
    n, converged = newton_solver.solve(wh)
    print(
        f"Newton solver for The Navier-Stokes equations completed in {n} iterations. Converged: {converged}"
    )

    # --- Extract and sort pressure on the top face (z = Lz) ---
    pressure_field = wh.sub(1).collapse()

    dof_coordinates = pressure_field.function_space.tabulate_dof_coordinates()

    top_wall_dofs = dolfinx.fem.locate_dofs_topological(
        pressure_field.function_space, fluid_domain.topology.dim - 1, top_facet)

    top_wall_coords = dof_coordinates[top_wall_dofs]

    interface_pressure_values = pressure_field.x.array[top_wall_dofs]

    sorted_indices = np.argsort(top_wall_coords[:, 0])
    interface_pressure = interface_pressure_values[sorted_indices]

    sorted_interface_pressure_x = np.empty_like(interface_pressure)

    for j in range(n_y + 1):
        sorted_interface_pressure_x[j * (n_x + 1):(j + 1) * (n_x + 1)] = interface_pressure[
            (n_x - np.arange(n_x + 1)) * (n_y + 1) + j]

    return wh, sorted_interface_pressure_x

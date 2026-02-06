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
        pressure_field.function_space, fluid_domain.topology.dim - 1, top_facet
    )

    top_wall_coords = dof_coordinates[top_wall_dofs]

    interface_pressure_values = pressure_field.x.array[top_wall_dofs]

    sorted_indices = np.argsort(top_wall_coords[:, 0])
    interface_pressure = interface_pressure_values[sorted_indices]

    sorted_interface_pressure_x = np.empty_like(interface_pressure)

    for j in range(n_y + 1):
        sorted_interface_pressure_x[j * (n_x + 1) : (j + 1) * (n_x + 1)] = (
            interface_pressure[(n_x - np.arange(n_x + 1)) * (n_y + 1) + j]
        )

    # -------------------------------------------------------------------------
    # Volumetric flow rate at outlet: Q_outlet = ∫_{Γ_out} u · n ds
    # -------------------------------------------------------------------------

    # We need a facet tag for the outlet to build a boundary measure restricted
    # to Γ_out. We tag the outlet facets with ID = 1.
    fdim = fluid_domain.topology.dim - 1

    # tag outlet facets with ID = 1
    outlet_values = np.full(len(outlet_facet), 1, dtype=np.int32)
    outlet_tags = dolfinx.mesh.meshtags(fluid_domain, fdim,
                                        outlet_facet, outlet_values)

    ds_out = ufl.Measure("ds", domain=fluid_domain, subdomain_data=outlet_tags)

    # Use the solved velocity uh
    n_normal = ufl.FacetNormal(fluid_domain)
    Q_form = dolfinx.fem.form(ufl.dot(uh, n_normal) * ds_out(1))

    Q_local = dolfinx.fem.assemble_scalar(Q_form)
    Q_outlet = fluid_domain.comm.allreduce(Q_local, op=MPI.SUM)

    # Note: with outward normal n, a positive Q_outlet corresponds to flow
    # leaving the domain through the outlet.

    return wh, sorted_interface_pressure_x, Q_outlet


# -----------------------------------------------------------------------------
# Steady 3D Navier–Stokes solver for a composite channel with a rigid–elastic–rigid
# configuration and pressure boundary conditions
# -----------------------------------------------------------------------------
#
# Geometry:
#   - A 3D channel in the x-direction with three parts:
#       * Left channel      : rigid walls
#       * Middle channel    : top wall is elastic (Kirchhoff–Love plate) and the rests are rigid
#       * Right channel     : rigid walls
#
#   The domain is described by:
#       - x ∈ [0, channel_length]
#       - y ∈ [0, channel_width]
#       - z ∈ [0, channel_height]
#
#   The left and right sub-channels are identified via x_min_channel_right and
#   x_max_channel_right. The remaining middle part is the elastic-top section.
#
# PDE model:
#   - Steady incompressible Navier–Stokes equations (non-dimensional form):
#
#       (u · ∇)u - (1/Re) Δu + ∇p = 0     in Ω
#                           ∇·u = 0       in Ω
#
#   - Boundary conditions:
#       * Inlet  (x = x_max_channel_left)  : prescribed pressure = inlet_pressure
#       * Outlet (x = x_min_channel_right) : prescribed pressure = outlet_pressure
#       * All walls (including top)        : no-slip (u = 0)
#
# Purpose:
#   This function solves the steady Navier–Stokes problem in the given mesh
#   and extracts the pressure distribution on the *top middle wall* (the elastic
#   section). This pressure distribution is reorganized into a 1D array
#   `top_middle_wall_p_sorted_x`, arranged so that it can be used as a load
#   distribution (source term) in a Kirchhoff–Love plate bending model to compute
#   the deflection of the elastic top wall.
#
# Output:
#   - wh                         : mixed solution Function (u, p)
#   - top_middle_wall_p_sorted_x : pressure on the elastic top wall, reordered
#                                  consistently along x for each y
#   - channel_length_middle      : used as the plate length
#   - unique_x                   : sorted unique x-coordinates on that wall
#   - unique_y                   : sorted unique y-coordinates on that wall
# -----------------------------------------------------------------------------
def pressure_3d_pressure_inlet_rigid_elastic_rigid_channel(
    fluid_domain: dolfinx.mesh.Mesh,
    channel_length: float,
    channel_width: float,
    channel_height: float,
    x_max_channel_right: float,
    x_min_channel_left: float,
    inlet_pressure: float,
    outlet_pressure: float,
    reynolds_number: float,
):
    """
    Solve steady 3D Navier–Stokes flow in a composite rigid–elastic–rigid channel
    and extract the pressure distribution on the elastic top wall.

    The function assumes a straight 3D channel aligned with the x-axis, composed of
    three parts:
        - Left section  (rigid top wall)
        - Middle section (elastic top wall, modeled later as a Kirchhoff–Love plate)
        - Right section (rigid top wall)

    The flow is driven by a prescribed pressure difference between inlet and outlet
    (pressure boundary conditions), while all walls (including the elastic wall)
    are no-slip for the fluid.

    Parameters
    ----------
    fluid_domain :
        dolfinx.mesh.Mesh representing the full fluid domain.
    channel_length : float
        Total length of the channel in the x-direction.
    channel_width : float
        Channel width in the y-direction (domain extent in y).
    channel_height : float
        Channel height in the z-direction (domain extent in z).
    x_max_channel_right : float
        x-coordinate of the right end of the right channel section.
    x_min_channel_left : float
        x-coordinate of the left end of the left channel section.
    inlet_pressure : float
        Prescribed pressure at the inlet boundary (x = x_max_channel_left).
    outlet_pressure : float
        Prescribed pressure at the outlet boundary (x = x_min_channel_right).
    reynolds_number : float
        Reynolds number Re = (U L) / ν used to non-dimensionalize the equations.
    Returns
    -------
    wh :
        dolfinx.fem.Function representing the mixed solution (u, p)
    top_middle_wall_p_sorted_x : numpy.ndarray
        1D array of pressure values on the elastic top middle wall, reordered
        so that for each y-level the values are sorted along x. This array is
        intended to be used as the load/source function for the Kirchhoff–Love
        plate bending problem.
    channel_length_middle: float
        used as the plate length.
    unique_x : numpy.ndarray
        Sorted array of unique x-coordinates on the top middle wall.
        used as number of cells (or divisions) in x-direction in the plate mesh.
    unique_y : numpy.ndarray
        Sorted array of unique y-coordinates on the top middle wall.
        used as number of cells (or divisions) in y-direction in the plate mesh.
    """

    # -------------------------------------------------------------------------
    # 1. Geometric partition of the channel into left / middle / right sections
    # -------------------------------------------------------------------------

    # For simplicity, we reset the left and right x-limits of the rigid sections.
    x_min_channel_right = 0
    x_max_channel_left = channel_length
    # Middle section is defined as the interval between the left and right sections
    x_min_channel_middle = x_max_channel_right
    x_max_channel_middle = x_min_channel_left
    # Lengths of the left, middle and right sections (useful for diagnostics)
    channel_length_middle = x_max_channel_middle - x_min_channel_middle
    # channel_length_left = x_max_channel_left - x_min_channel_left
    # channel_length_right = x_max_channel_right - x_min_channel_right

    # -------------------------------------------------------------------------
    # 2. Mixed finite element spaces: velocity (vector P2) and pressure (scalar P1)
    # -------------------------------------------------------------------------
    el_u = basix.ufl.element(
        "Lagrange",
        fluid_domain.topology.cell_name(),
        2,
        shape=(fluid_domain.geometry.dim,),
    )
    el_p = basix.ufl.element("Lagrange", fluid_domain.topology.cell_name(), 1)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])
    # Mixed function space W = [P2]^d × P1
    W = dolfinx.fem.functionspace(fluid_domain, el_mixed)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    # Boundary measure and facet normal
    ds = ufl.Measure("ds", domain=fluid_domain)
    n_normal = ufl.FacetNormal(fluid_domain)
    # Velocity subspace (W0) and its collapsed scalar/vector space (V)
    W0 = W.sub(0)
    V, _ = W0.collapse()
    # Mixed solution (u, p) as a nonlinear function
    wh = dolfinx.fem.Function(W)
    uh, ph = ufl.split(wh)

    # -------------------------------------------------------------------------
    # 3. Locate boundary facets for inlet, outlet and walls
    # -------------------------------------------------------------------------
    # Inlet: plane x = x_max_channel_left
    def inlet_marker(x):
        return np.isclose(x[0], x_max_channel_left)
    inlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, inlet_marker
    )

    # Outlet: plane x = x_min_channel_right
    def outlet_marker(x):
        return np.isclose(x[0], x_min_channel_right)
    outlet_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, outlet_marker
    )

    # All other walls except the top (z = channel_height):
    #   y = 0, y = channel_width
    def walls_sides(x):
        return (
                np.isclose(x[1], 0)
                | np.isclose(x[1], channel_width)
        )
    walls_sides_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, walls_sides
    )

    tol = 1e-8
    # Top wall in the left rigid section: z = channel_height and x in (left section)
    def wall_top_left_channel(x):
        return np.logical_and.reduce((
            np.isclose(x[2], channel_height, atol=tol),
            x[0] > x_min_channel_left - tol,
            x[0] <= x_max_channel_left + tol
        ))
    wall_top_left_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, wall_top_left_channel
    )

    def wall_top_right_channel(x):
        return np.logical_and.reduce((
            np.isclose(x[2], channel_height, atol=tol),
            x[0] >= x_min_channel_right - tol,
            x[0] < x_max_channel_right + tol
        ))
    wall_top_right_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, wall_top_right_channel
    )

    def wall_bottom_left_channel(x):
        return np.logical_and.reduce((
            np.isclose(x[2], 0, atol=tol),
            x[0] > x_min_channel_left - tol,
            x[0] <= x_max_channel_left + tol
        ))
    wall_bottom_left_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, wall_bottom_left_channel
    )

    def wall_bottom_right_channel(x):
        return np.logical_and.reduce((
            np.isclose(x[2], 0, atol=tol),
            x[0] >= x_min_channel_right - tol,
            x[0] < x_max_channel_right + tol
        ))
    wall_bottom_right_facet = dolfinx.mesh.locate_entities_boundary(
        fluid_domain, fluid_domain.topology.dim - 1, wall_bottom_right_channel
    )

    # All exterior facets of the domain
    all_boundary_facets = dolfinx.mesh.exterior_facet_indices(fluid_domain.topology)
    # middle walls facets are the remaining facets on z = channel_height that
    # are not part of inlet, outlet, lower/side walls, or top-left/right.
    walls_middle_facet = np.setdiff1d(
        all_boundary_facets,
        np.unique(
            np.concatenate(
                (
                    inlet_facet,
                    outlet_facet,
                    walls_sides_facet,
                    wall_top_left_facet,
                    wall_top_right_facet,
                    wall_bottom_left_facet,
                    wall_bottom_right_facet,
                )
            )
        ),
    )
    # -------------------------------------------------------------------------
    # 4. Create Dirichlet BCs for pressure (inlet/outlet) and velocity (walls)
    # -------------------------------------------------------------------------

    # Pressure: locate dofs on inlet and outlet facets in W.sub(1) (pressure space)
    dofs_inlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, inlet_facet
    )
    dofs_outlet = dolfinx.fem.locate_dofs_topological(
        W.sub(1), fluid_domain.topology.dim - 1, outlet_facet
    )

    # Velocity: locate dofs on the rigid walls in (W0, V) pair
    dofs_walls_sides = dolfinx.fem.locate_dofs_topological((W0, V), fluid_domain.topology.dim - 1, walls_sides_facet)

    dofs_wall_top_left = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, wall_top_left_facet)
    dofs_wall_top_right = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, wall_top_right_facet)
    dofs_wall_bottom_left = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, wall_bottom_left_facet)
    dofs_wall_bottom_right = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, wall_bottom_right_facet)

    dofs_walls_middle = dolfinx.fem.locate_dofs_topological(
        (W0, V), fluid_domain.topology.dim - 1, walls_middle_facet)

    # Inlet pressure BC
    p_inlet = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(inlet_pressure)
    )
    bc_inlet = dolfinx.fem.dirichletbc(p_inlet, dofs_inlet, W.sub(1))

    # Outlet pressure BC
    p_outlet = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(outlet_pressure)
    )
    bc_outlet = dolfinx.fem.dirichletbc(p_outlet, dofs_outlet, W.sub(1))

    # No-slip velocity BC on all walls (including top sections)
    u_wall = dolfinx.fem.Function(V)
    u_wall.x.array[:] = 0

    bc_walls_sides = dolfinx.fem.dirichletbc(u_wall, dofs_walls_sides, W0)
    bc_wall_top_left = dolfinx.fem.dirichletbc(u_wall, dofs_wall_top_left, W0)
    bc_wall_top_right = dolfinx.fem.dirichletbc(u_wall, dofs_wall_top_right, W0)
    bc_wall_bottom_left = dolfinx.fem.dirichletbc(u_wall, dofs_wall_bottom_left, W0)
    bc_wall_bottom_right = dolfinx.fem.dirichletbc(u_wall, dofs_wall_bottom_right, W0)
    bc_walls_middle = dolfinx.fem.dirichletbc(u_wall, dofs_walls_middle, W0)

    # Collect all boundary conditions
    bcs = [bc_inlet,
           bc_outlet,
           bc_walls_sides,
           bc_wall_top_left,
           bc_wall_top_right,
           bc_wall_bottom_left,
           bc_wall_bottom_right,
           bc_walls_middle,
           ]

    # -------------------------------------------------------------------------
    # 5. Define the weak form of the steady Navier–Stokes equations
    # -------------------------------------------------------------------------

    Re = dolfinx.fem.Constant(
        fluid_domain, dolfinx.default_scalar_type(reynolds_number)
    )

    # --- Weak form of steady NS (do-nothing form for pressure BCs via +∫ p n·v ds) ---
    F = ufl.inner(ufl.grad(uh) * uh, v) * ufl.dx  # Convective term
    F += ((1 / Re) * ufl.inner(ufl.grad(uh), ufl.grad(v))) * ufl.dx  # Diffusion term
    F -= ufl.inner(ph, ufl.div(v)) * ufl.dx  # Pressure gradient
    F += ufl.dot(ph * n_normal, v) * ds  # Weak imposition of Dirichlet conditions
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

    # -------------------------------------------------------------------------
    # 7. Identify the top middle (elastic) wall and extract pressure data
    # -------------------------------------------------------------------------
    def wall_top_middle_channel(x):
        """Logical mask for vertices on the top middle (elastic) wall."""
        return np.logical_and.reduce(
            (
                np.isclose(x[2], channel_height, atol=tol),
                x[0] >= x_min_channel_middle - tol,
                x[0] <= x_max_channel_middle + tol,
            )
        )

    # Vertices belonging to the top middle wall
    verts = dolfinx.mesh.locate_entities(fluid_domain, 0, wall_top_middle_channel)
    # Coordinates of those vertices
    coords = fluid_domain.geometry.x[verts]

    # Unique x and y coordinates on the top middle wall (rounded to avoid noise)
    unique_x = np.unique(np.round(coords[:, 0], 10))
    unique_y = np.unique(np.round(coords[:, 1], 10))

    # -------------------------------------------------------------------------
    # 8. Extract and sort pressure on the top middle face
    # -------------------------------------------------------------------------
    # Collapse the pressure subspace to obtain a scalar pressure field
    pressure_field = wh.sub(1).collapse()
    # Coordinates of pressure dofs
    dof_coordinates = pressure_field.function_space.tabulate_dof_coordinates()
    # Find pressure dofs that belong to the top middle wall facets
    middle_walls_dofs = dolfinx.fem.locate_dofs_topological(
        pressure_field.function_space, fluid_domain.topology.dim - 1, walls_middle_facet)
    # Coordinates and pressure values at those dofs
    middle_walls_coords = dof_coordinates[middle_walls_dofs]
    middle_walls_p = pressure_field.x.array[middle_walls_dofs]
    # Sort by x-coordinate (for now)
    sorted_indices = np.argsort(middle_walls_coords[:, 0])
    middle_walls_p_sorted = middle_walls_p[sorted_indices]
    new_middle_walls_p_sorted = middle_walls_p_sorted.reshape(-1, 2 * len(unique_y))[:, :len(unique_y)].ravel()

    # Reorder pressures into a consistent x-sweep for each fixed y, to match
    # the logical (n_x+1) × (n_y+1) grid on the top wall.
    middle_walls_p_sorted_x = np.empty_like(new_middle_walls_p_sorted)

    n_x = len(unique_x) - 1
    n_y = len(unique_y) - 1
    for j in range(n_y + 1):
        middle_walls_p_sorted_x[j * (n_x + 1):(j + 1) * (n_x + 1)] = new_middle_walls_p_sorted[
            (n_x - np.arange(n_x + 1)) * (n_y + 1) + j]

    # This pressure array is now suitable as a load distribution for a
    # Kirchhoff–Love plate bending model of the elastic top wall.

    # -------------------------------------------------------------------------
    # 9. Volumetric flow rate at outlet: Q_outlet = ∫_{Γ_out} u · n ds
    # -------------------------------------------------------------------------

    # We need a facet tag for the outlet to build a boundary measure restricted
    # to Γ_out. We tag the outlet facets with ID = 1.
    fdim = fluid_domain.topology.dim - 1

    # tag outlet facets with ID = 1
    outlet_values = np.full(len(outlet_facet), 1, dtype=np.int32)
    outlet_tags = dolfinx.mesh.meshtags(fluid_domain, fdim,
                                        outlet_facet, outlet_values)

    ds_out = ufl.Measure("ds", domain=fluid_domain, subdomain_data=outlet_tags)

    # Use the solved velocity uh
    Q_form = dolfinx.fem.form(ufl.dot(uh, n_normal) * ds_out(1))

    Q_local = dolfinx.fem.assemble_scalar(Q_form)
    Q_outlet = fluid_domain.comm.allreduce(Q_local, op=MPI.SUM)

    # Note: with outward normal n, a positive Q_outlet corresponds to flow
    # leaving the domain through the outlet.

    return wh, Q_outlet, middle_walls_p_sorted_x, channel_length_middle, unique_x, unique_y

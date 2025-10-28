from mpi4py import MPI
import dolfinx
import basix.ufl
import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver



def pressure_2d_velocity_inlet(
    fluid_domain,
    reynolds_number: float,
):
    # Finite elements and mixed function space
    el_u = basix.ufl.element("Lagrange", fluid_domain.basix_cell(), 2, shape=(2,))
    el_p = basix.ufl.element("Lagrange", fluid_domain.basix_cell(), 1)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])
    # Test and trial functions in mixed spaces
    W = dolfinx.fem.functionspace(fluid_domain, el_mixed)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    # Functions
    wh = dolfinx.fem.Function(W)
    uh, ph = ufl.split(wh)
    # Variational form
    Re = dolfinx.fem.Constant(fluid_domain, dolfinx.default_scalar_type(reynolds_number))

    F = Re * ufl.inner(ufl.dot(ufl.grad(uh), uh), v) * ufl.dx  # Convective term
    F += ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx  # Diffusion term
    F -= ufl.inner(ph, ufl.div(v)) * ufl.dx  # Pressure gradient
    F -= ufl.inner(q, ufl.div(uh)) * ufl.dx  # Continuity equation

    # Locating a subset of entities on a boundary
    def inlet_marker(x):
        return np.isclose(x[0], 0.0)

    fluid_domain.topology.create_connectivity(fluid_domain.topology.dim - 1, fluid_domain.topology.dim)
    inlet_facets = dolfinx.mesh.locate_entities_boundary(fluid_domain, fluid_domain.topology.dim - 1, inlet_marker)

    def top_bottom_marker(x):
        return np.isclose(x[1], 1.0) | np.isclose(x[1], 0.0)

    tb_facets = dolfinx.mesh.locate_entities_boundary(fluid_domain, fluid_domain.topology.dim - 1, top_bottom_marker)

    all_boundary_facets = dolfinx.mesh.exterior_facet_indices(fluid_domain.topology)
    outlet_facets = np.setdiff1d(all_boundary_facets, np.union1d(tb_facets, inlet_facets))

    # Dirichlet conditions in mixed spaces
    W0 = W.sub(0)
    V, V_to_W0 = W0.collapse()

    u_inlet = dolfinx.fem.Function(V)
    class InletVelocity:
        def __call__(self, x):
            value = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            value[0] = 6 * x[1] * (1 - x[1])
            return value
    parabolic_velocity = InletVelocity()
    u_inlet.interpolate(parabolic_velocity)
    dofs_inlet = dolfinx.fem.locate_dofs_topological((W0, V), fluid_domain.topology.dim - 1, inlet_facets)
    bc_inlet = dolfinx.fem.dirichletbc(u_inlet, dofs_inlet, W0)

    p_outlet = dolfinx.fem.Constant(fluid_domain, dolfinx.default_scalar_type(0))
    dofs_outlet = dolfinx.fem.locate_dofs_topological(W.sub(1), fluid_domain.topology.dim - 1, outlet_facets)
    bc_outlet = dolfinx.fem.dirichletbc(p_outlet, dofs_outlet, W.sub(1))

    u_wall = dolfinx.fem.Function(V)
    u_wall.x.array[:] = 0
    dofs_wall = dolfinx.fem.locate_dofs_topological((W0, V), fluid_domain.topology.dim - 1, tb_facets)
    bc_wall = dolfinx.fem.dirichletbc(u_wall, dofs_wall, W0)

    bcs = [bc_inlet, bc_outlet, bc_wall]

    # Create the nonlinear problem
    problem = NonlinearProblem(F, wh, bcs)

    # Create the Newton solver
    newton_solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Set the Newton solver parameters
    newton_solver.convergence_criterion = "incremental"
    newton_solver.rtol = 1e-8
    newton_solver.max_it = 1000
    newton_solver.report = True

    # Modify the linear solver in each Newton iteration
    ksp = newton_solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    # Solve the nonlinear problem
    n, converged = newton_solver.solve(wh)
    print(f"Newton solver completed in {n} iterations. Converged: {converged}")


    return wh


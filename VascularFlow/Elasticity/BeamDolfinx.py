import basix
import numpy as np
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
import ufl


def euler_bernoulli_transient_dolfinx(
    nb_nodes, time_step_size, nb_time_steps, load, beta, relaxation, H_new
):
    """
    calculate the deflection of a beam under the Euler Bernoulli beam theory using FEniCSx

    parameters:
    ----------
    nb_nodes : int
        The number of positional nodes along the beam.
    time_step_size : float
        The time step size of the simulation.
    end_time : float
        The end time of the simulation.
    load : np.ndarray
        The load applied to the beam.

    returns:
    ----------
    deflection_n : np.array
        The deflection of the beam for each nodal position along the beam.

    """

    domain = mesh.create_unit_interval(MPI.COMM_WORLD, nb_nodes)
    beam_element = basix.ufl.element(
        basix.ElementFamily.Hermite, basix.CellType.interval, 3
    )
    W = functionspace(domain, beam_element)
    w = ufl.TrialFunction(W)
    w_test_function = ufl.TestFunction(W)

    # Boundary conditions
    boundary_condition_function = fem.Function(W)

    start_point = locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], 0))
    end_point = locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], 1))
    start_dof = fem.locate_dofs_topological(W, 0, start_point)
    end_dof = fem.locate_dofs_topological(W, 0, end_point)

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

    # initialization
    w_n = fem.Function(W)
    w_n_1 = fem.Function(W)

    P = fem.Function(W)
    p_interleaved = np.zeros(len(load) * 2)
    p_interleaved[::2] = load
    P.x.array[:] = p_interleaved

    # Variational form
    lhs = (1 / (time_step_size**2)) * ufl.inner(w, w_test_function) * ufl.dx + ufl.dot(
        ufl.div(ufl.grad(w)), ufl.div(ufl.grad(w_test_function))
    ) * ufl.dx
    rhs = (
        P * w_test_function * ufl.dx
        + (2 / (time_step_size**2)) * ufl.inner(w_n, w_test_function) * ufl.dx
        - (1 / (time_step_size**2)) * ufl.inner(w_n_1, w_test_function) * ufl.dx
    )

    # Store results for animation
    displacement_history = []
    # time-stepping loop
    for n in range(nb_time_steps):
        time = n * time_step_size

        # solve linear problem
        problem = LinearProblem(
            lhs, rhs, bcs=BCs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        w = problem.solve()

        # update solution for next time step
        w_n_1.x.array[:] = w_n.x.array
        w_n.x.array[:] = w.x.array
        displacement = np.array([w.x.array[i] for i in range(0, len(w.x.array), 2)])
        displacement_history.append(displacement.copy())

    height = 1 + (beta * displacement)
    height = relaxation * height + (1 - relaxation) * H_new

    return height

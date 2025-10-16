"""

Compare the 1D numerical and analytical solutions of the linear Euler–Bernoulli beam equation
for an elastic beam of length 1, subjected to a constant distributed load applied at each nodal position.

"""

import numpy as np
import pytest

from VascularFlow.Elasticity.Beam import (
    euler_bernoulli_steady,
    euler_bernoulli_transient,
)


@pytest.mark.parametrize(
    "nb_mesh_nodes, constant_load",
    [
        (51, 1),
    ],
)
def test_euler_bernoulli_steady_constant_load(nb_mesh_nodes, constant_load, plot=True):
    left = 0
    right = 50
    mesh_nodes = np.linspace(left, right, nb_mesh_nodes)
    distributed_load = np.full(nb_mesh_nodes, constant_load)
    disp, lhs, rhs = euler_bernoulli_steady(mesh_nodes, distributed_load)
    #print(lhs)
    #print(rhs)

    x_n = mesh_nodes
    # analytical solution for ∂4w/∂x4 = constant load
    disp_exact = (
        (constant_load / 24) * x_n**4
        - (constant_load / 12) * x_n**3
        + (constant_load / 24) * x_n**2
    )
    #np.testing.assert_allclose(disp, disp_exact, rtol=1e-4, atol=1e-8)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(mesh_nodes[0:5], disp[0:5], "+", label="numerical solution")
        #plt.plot(mesh_nodes, disp_exact, "*", label="exact solution")
        plt.xlabel("x")
        plt.ylabel("displacement")
        plt.legend()
        plt.title("Euler Bernoulli steady, numerical vs analytical")
        plt.show()


def test_euler_bernoulli_constant_load_transient(plot=True):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, 101)

    time_step_size = 2.5e-03
    nb_time_steps = 2000
    distributed_load = np.ones(len(mesh_nodes))
    beta = 35156.24
    relaxation_factor = 0.00003
    h_new = np.ones(len(mesh_nodes))

    channel_height = euler_bernoulli_transient(
        mesh_nodes,
        nb_time_steps,
        time_step_size,
        distributed_load,
        beta,
        relaxation_factor,
        h_new,
    )

    x_n = mesh_nodes
    # analytical solution for ∂4w/∂x4 = 1
    disp_exact = (1 / 24) * x_n**4 - (1 / 12) * x_n**3 + (1 / 24) * x_n**2
    # update channel height
    channel_height_exact = 1 + beta * disp_exact
    channel_height_exact = (
        relaxation_factor * channel_height_exact + (1 - relaxation_factor) * h_new
    )
    np.testing.assert_allclose(
        channel_height_exact, channel_height, rtol=1e-4, atol=1e-8
    )
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(mesh_nodes, channel_height, "+", label="numerical solution")
        plt.plot(mesh_nodes, channel_height_exact, "*", label="exact solution")
        plt.xlabel("x")
        plt.ylabel("displacement")
        plt.legend()
        plt.title("Euler Bernoulli transient, numerical vs analytical")
        plt.show()

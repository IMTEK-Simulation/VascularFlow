"""
Test the two-way coupled fluid–structure interaction (FSI) solver across different simulation configurations.

This test verifies:
- The correct shape of the computed pressure, flow rate, and channel height arrays.
- That the FSI solver runs without errors for a range of physical and numerical parameters.

The solver couples:
- Navier–Stokes equations (for pressure and flow rate)
- Euler–Bernoulli beam equation (for wall displacement)

Boundary conditions include:
- Inlet flow rate = 1
- Outlet pressure = 0
- Clamped beam ends (zero displacement and rotation)

No analytical solution is available, so correctness is based on shape and execution success.
"""

import numpy as np
import pytest

from VascularFlow.Coupling.OneDimensional import two_way_coupled_fsi


@pytest.mark.parametrize(
    "nb_nodes, time_step_size, channel_aspect_ratio, reynolds_number, strouhal_number, "
    "fsi_parameter, relaxation_factor, inner_res, inner_it_number",
    [(101, 1e-03, 0.02, 7.5, 0.68, 35156.24, 0.001, 1e-4, 2000)],
)
def test_two_way_coupled_fsi(
    nb_nodes,
    time_step_size,
    channel_aspect_ratio,
    reynolds_number,
    strouhal_number,
    fsi_parameter,
    relaxation_factor,
    inner_res,
    inner_it_number,
    plot=True,
):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, nb_nodes)
    nb_time_steps = 10
    end_time = nb_time_steps * time_step_size
    inlet_flow_rate = 1
    inner_tolerance = 1e-16

    # Initialize flow arrays
    h_n_1 = np.ones(nb_nodes)
    h_n = np.ones(nb_nodes)
    h_star = np.ones(nb_nodes)
    h_new = np.ones(nb_nodes)
    q_n_1 = np.ones(nb_nodes)
    q_n = np.ones(nb_nodes)
    q_star = np.ones(nb_nodes)
    q_new = np.ones(nb_nodes)
    p = np.zeros(nb_nodes)
    p_new = np.zeros(nb_nodes)

    h_n, q_n, p, residual_values, iteration_indices = two_way_coupled_fsi(
        mesh_nodes,
        time_step_size,
        end_time,
        channel_aspect_ratio,
        reynolds_number,
        strouhal_number,
        fsi_parameter,
        relaxation_factor,
        inlet_flow_rate,
        inner_res,
        inner_it_number,
        inner_tolerance,
        h_n_1,
        h_n,
        h_star,
        h_new,
        q_n_1,
        q_n,
        q_star,
        q_new,
        p,
        p_new,
    )

    # Basic assertions
    assert h_n.shape == (nb_nodes,)
    assert q_n.shape == (nb_nodes,)
    assert p.shape == (nb_nodes,)

    print(h_n)
    print(q_n)
    print(p)

    # Optional plot
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.semilogy(iteration_indices, residual_values)
        plt.xlabel("Cumulative Inner Iteration Count")
        plt.ylabel("Inner Residual (log scale)")
        plt.title("Inner Residual vs Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Inner Residual vs Iterations.png")
        plt.show()

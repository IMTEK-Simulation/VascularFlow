import numpy as np
import pytest

from VascularFlow.Coupling.OneDimensionalSteady import steady_state_fsi


@pytest.mark.parametrize(
    "nb_nodes, channel_aspect_ratio, reynolds_number,"
    "fsi_parameter, relaxation_factor, residual_number, iteration_number",
    [(4, 0.02, 7.5, 35156.24, 0.0003, 1e-3, 1)],
)
def test_two_way_coupled_fsi_steady(
    nb_nodes,
    channel_aspect_ratio,
    reynolds_number,
    fsi_parameter,
    relaxation_factor,
    residual_number,
    iteration_number,
    plot=True,
):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, nb_nodes)
    inner_tolerance = 1e-16

    # Initialize flow arrays
    h_star = np.ones(nb_nodes)
    h_new = np.ones(nb_nodes)

    p_new = np.zeros(nb_nodes)

    pressure, height = steady_state_fsi(
        mesh_nodes,
        channel_aspect_ratio,
        reynolds_number,
        fsi_parameter,
        relaxation_factor,
        inner_tolerance,
        residual_number,
        iteration_number,
        h_star,
        h_new,
        p_new,
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(mesh_nodes, pressure)
        ax[1].plot(mesh_nodes, height)

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("P")

        ax[1].set_xlabel("x")
        ax[1].set_ylabel("H")

        plt.tight_layout()
        plt.show()

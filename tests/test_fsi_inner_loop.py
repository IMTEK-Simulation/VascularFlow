"""
Test the inner_fsi_iteration function used in the 1D fluid–structure interaction (FSI) solver.

This test returns updated pressure, flow rate, and channel height displacement
"""

import numpy as np
import pytest
from VascularFlow.Coupling.FSIInnerLoop import inner_fsi_iteration


@pytest.mark.parametrize(
    "nb_nodes, channel_aspect_ratio, reynolds_number, strouhal_number, fsi_parameter, relaxation_factor, time_step_size, inner_tolerance",
    [
        (200, 0.02, 7.5, 0.68, 35156.24, 0.00003, 2.5e-3, 1e-20),
    ],
)
def test_inner_fsi_iteration(
    nb_nodes,
    channel_aspect_ratio,
    reynolds_number,
    strouhal_number,
    fsi_parameter,
    relaxation_factor,
    time_step_size,
    inner_tolerance,
):
    mesh_nodes = np.linspace(0, 1, nb_nodes)
    h_n_1 = np.ones(nb_nodes)
    h_n = np.ones(nb_nodes)
    h_star = np.ones(nb_nodes)
    h_new = np.ones(nb_nodes)
    q_n_1 = np.ones(nb_nodes)
    q_n = np.ones(nb_nodes)
    q_star = np.ones(nb_nodes)
    p_new = np.zeros(nb_nodes)

    h_star, q_star, p, inner_residual = inner_fsi_iteration(
        mesh_nodes,
        time_step_size,
        channel_aspect_ratio,
        reynolds_number,
        strouhal_number,
        fsi_parameter,
        relaxation_factor,
        1,
        inner_tolerance,
        h_n_1,
        h_n,
        h_star,
        h_new,
        q_n_1,
        q_n,
        q_star,
        p_new,
    )

    assert h_star.shape == (nb_nodes,)
    assert q_star.shape == (nb_nodes,)
    assert p.shape == (nb_nodes,)



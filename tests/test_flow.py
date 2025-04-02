"""
Test the `flow_rate` function under varying mesh sizes and sinusoidal channel height distributions.

Since no analytical solution is available, this test ensures:
- Output shape is consistent with input mesh.
- Output contains no NaNs or infinities.
"""

import numpy as np
import pytest
from VascularFlow.Flow.Flow import flow_rate


@pytest.mark.parametrize(
    "nb_nodes, time_step_size, h_star",
    [
        (11, 0.01, 0.0001 * np.sin(np.pi * np.linspace(0, 1, 11)) + 1),
        (21, 0.005, 0.00002 * np.sin(np.pi * np.linspace(0, 1, 21)) + 1),
        (51, 0.0025, 0.00003 * np.sin(np.pi * np.linspace(0, 1, 51)) + 1),
    ],
)
def test_flow_rate(nb_nodes, time_step_size, h_star, plot=True):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, nb_nodes)
    st = 0.68
    inlet_flow_rate = 1

    h_n = np.ones(nb_nodes)
    h_n1 = np.ones(nb_nodes)

    channel_flow_rate = flow_rate(
        mesh_nodes, time_step_size, st, inlet_flow_rate, h_star, h_n, h_n1
    )

    # 1. Assert shape is correct
    assert channel_flow_rate.shape == (nb_nodes,)
    # 2. Assert no NaNs or infs
    assert np.all(np.isfinite(channel_flow_rate))

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(
            mesh_nodes, channel_flow_rate, label=f"St={st}, Inlet={inlet_flow_rate}"
        )
        plt.xlabel("x")
        plt.ylabel("Area Flow Rate")
        plt.grid(True)
        plt.title("Computed Area Flow Rate")
        plt.legend()
        plt.show()

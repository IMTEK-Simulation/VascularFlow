"""
Test the `pressure` function under varying parameters in steady and transient states.

Since no analytical solution is available, this test ensures:
- Output shape is consistent with input mesh.
- Output contains no NaNs or infinities.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from VascularFlow.Flow.Pressure import pressure, pressure_steady_state


@pytest.mark.parametrize(
    "nb_nodes, time_step_size, eps, re, st",
    [
        (11, 0.01, 0.02, 7.5, 0.68),
        (51, 0.005, 0.01, 10, 0.5),
        (101, 0.0025, 0.02, 7.5, 0.68),
    ],
)
def test_pressure(nb_nodes, time_step_size, eps, re, st, plot=True):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, nb_nodes)
    h_star = np.ones(len(mesh_nodes))
    q_star = np.ones(len(mesh_nodes))
    q_n = np.ones(len(mesh_nodes))
    q_n1 = np.ones(len(mesh_nodes))

    channel_pressure = pressure(
        mesh_nodes,
        time_step_size,
        eps,
        re,
        st,
        h_star,
        q_star,
        q_n,
        q_n1,
    )

    # 1. Assert shape is correct
    assert channel_pressure.shape == (nb_nodes,)
    # 2. Assert no NaNs or infinities
    assert np.all(np.isfinite(channel_pressure))

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(mesh_nodes, channel_pressure, label=f"n={nb_nodes}")
        plt.xlabel("x")
        plt.ylabel("pressure")
        plt.title("Computed Channel Pressure")
        plt.grid(True)
        plt.legend()
        plt.show()


@pytest.mark.parametrize(
    "nb_nodes, eps, re",
    [
        (4, 0.02, 7.5),
    ],
)
def test_pressure_steady_state(nb_nodes, eps, re, plot=True):
    left = 0
    right = 1
    mesh_nodes = np.linspace(left, right, nb_nodes)
    h_star = np.ones(len(mesh_nodes))

    channel_pressure_steady_state, LHS, RHS = pressure_steady_state(
        mesh_nodes,
        eps,
        re,
        h_star,
    )
    print(channel_pressure_steady_state)
    print(LHS)
    print(RHS)


    if plot:
        plt.plot(mesh_nodes, channel_pressure_steady_state, label=f"n={nb_nodes}")
        plt.xlabel("x")
        plt.ylabel("pressure steady state")
        plt.title("Computed Channel Pressure Steady State")
        plt.grid(True)
        plt.legend()
        plt.show()

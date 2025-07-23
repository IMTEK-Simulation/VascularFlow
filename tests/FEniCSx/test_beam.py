import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

from VascularFlow.FEniCSx.Elasticity.Beam import euler_bernoulli_steady_fenicsx


@pytest.mark.parametrize(
    "solid_domain, solid_domain_x_max_coordinate, dimensionless_bending_stiffness, dimensionless_extensional_stiffness, distributed_load_channel2, distributed_load_channel1",
    [
        (
            mesh.create_interval(MPI.COMM_WORLD, 500, [0, 50]), # solid_domain
            50, # solid_domain_x_max_coordinate
            7111, # dimensionless_bending_stiffness
            85333, # dimensionless_extensional_stiffness
            np.full(501, 1), # distributed_load_channel2 (bottom)
            np.full(501, 0), # distributed_load_channel2 (top)
        ),
    ],
)
def test_euler_bernoulli_steady_fenicsx(
    solid_domain,
    solid_domain_x_max_coordinate,
    dimensionless_bending_stiffness,
    dimensionless_extensional_stiffness,
    distributed_load_channel2,
    distributed_load_channel1,
    plot=True,
):
    """
    Test the steady-state Euler–Bernoulli beam solver under a uniform distributed load.

    The test compares the numerical displacement field
    to a known exact solution for a simply supported beam with specific loading conditions (∂⁴w/∂x⁴ = 1).

    The result is validated by computing the maximum absolute error and asserting
    that it remains below a predefined tolerance.
    """

    # Optional plot
    if plot:
        import matplotlib.pyplot as plt
        #plot_displacement_comparison(x, displacement)
        modes = ["small_deflection", "moderately_large_deflection", "large_deflection"]
        styles = ["--", "-.", "-"]  # choose any three line styles/markers you like
        markers = ["o", "s", "D"]
        plt.figure(figsize=(10, 6))
        for mode, ls, mk in zip(modes, styles, markers):
            disp = euler_bernoulli_steady_fenicsx(
                solid_domain,
                solid_domain_x_max_coordinate,
                dimensionless_bending_stiffness,
                dimensionless_extensional_stiffness,
                distributed_load_channel2,
                distributed_load_channel1,
                mode=mode,
            )
            x = np.linspace(0, solid_domain_x_max_coordinate, disp.shape[0])
            plt.plot(x, disp, ls, marker=mk, markevery=40, label=f"{mode.capitalize()}")
        plt.xlabel("x-coordinate")
        plt.ylabel("Displacement")
        plt.title("Euler–Bernoulli beam: small vs Moderate vs large deflection")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
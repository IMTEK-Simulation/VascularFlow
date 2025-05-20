import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

from VascularFlow.FEniCSx.Elasticity.Beam import euler_bernoulli_steady_fenicsx


@pytest.mark.parametrize(
    "solid_domain, solid_domain_x_max_coordinate, dimensionless_bending_stiffness, dimensionless_extensional_stiffness, distributed_load_channel2, distributed_load_channel1",
    [
        (
            mesh.create_interval(MPI.COMM_WORLD, 500, [0, 50]),
            50,
            1, # Bending stiffness
            0, # No axial stiffness
            np.ones(501), # Uniform load from channel 2 (top)
            np.full(501, 0), # No load from channel 1 (bottom)
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
    displacement = euler_bernoulli_steady_fenicsx(
        solid_domain,
        solid_domain_x_max_coordinate,
        dimensionless_bending_stiffness,
        dimensionless_extensional_stiffness,
        distributed_load_channel2,
        distributed_load_channel1,
        linera=True,
    )

    # Exact solution for comparison
    x = np.linspace(0, solid_domain_x_max_coordinate, displacement.shape[0])
    displacement_exact = (x ** 4 / 24) - ((25 / 6) * x ** 3) + ((625 / 6) * x ** 2)

    # Assertion: max absolute error between numerical and exact displacement
    max_error = np.max(np.abs(displacement - displacement_exact))
    assert max_error < 1e-2, f"Max displacement error too high: {max_error}"

    # Optional plot
    if plot:
        plot_displacement_comparison(x, displacement, displacement_exact)


def plot_displacement_comparison(x, displacement_numeric, displacement_exact):
    """
    Plot numerical vs. exact displacement profiles for visual comparison.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(x, displacement_numeric, "-o", label="Numerical")
    plt.plot(x, displacement_exact, "*", label="Exact")
    plt.xlabel("x-coordinate")
    plt.ylabel("Displacement")
    plt.title("Comparison of Numerical and Exact Displacement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

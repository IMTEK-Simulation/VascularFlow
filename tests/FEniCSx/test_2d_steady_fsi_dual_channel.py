import pytest
import numpy as np
from matplotlib import pyplot as plt


from VascularFlow.FEniCSx.Coupling.SteadyFSIDualChannel import (
    two_dimensional_steady_fsi_dual_channel,
)
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh


@pytest.mark.parametrize(
    (
        "fluid_solid_domain_length",
        "fluid_domain_height",
        "inlet_x_channel1",
        "outlet_x_channel1",
        "inlet_x_channel2",
        "outlet_x_channel2",
        "nb_mesh_nodes_x",
        "nb_mesh_nodes_y",
        "reynolds_number_channel1",
        "reynolds_number_channel2",
        "inlet_pressure_channel1",
        "inlet_pressure_channel2",
        "dimensionless_bending_stiffness",
        "dimensionless_extensional_stiffness",
        "under_relaxation_factor",
        "residual_number",
        "iteration_number",
    ),
    [
        # Format: (length, height, inlet1, outlet1, inlet2, outlet2, nx, ny, Re1, Re2, p1, p2, B, A, relax, tol, max_iter)
        (
            50,
            1,
            0,
            50,
            0,
            50,
            500,
            10,
            0.8,
            0.8,
            106,
            837,
            156250,
            2812500,
            0.0003,
            1e-4,
            1000,
        ),
    ],
)
def test_two_dimensional_steady_fsi_dual_channel(
    fluid_solid_domain_length,
    fluid_domain_height,
    inlet_x_channel1,
    outlet_x_channel1,
    inlet_x_channel2,
    outlet_x_channel2,
    nb_mesh_nodes_x,
    nb_mesh_nodes_y,
    reynolds_number_channel1,
    reynolds_number_channel2,
    inlet_pressure_channel1,
    inlet_pressure_channel2,
    dimensionless_bending_stiffness,
    dimensionless_extensional_stiffness,
    under_relaxation_factor,
    residual_number,
    iteration_number,
    epsilon=3e-16,
    plot_results=True,
):
    """
    Parametrized test of the 2D steady FSI dual-channel solver. Solves the coupled
    system and visualizes the deformed meshes and field profiles.
    """

    # Initial guesses
    w_new = np.zeros(nb_mesh_nodes_x + 1)
    p_new_1 = np.zeros(nb_mesh_nodes_x + 1)
    p_new_2 = np.zeros(nb_mesh_nodes_x + 1)

    # Run the FSI solver
    (
        channel1_pressure,
        channel2_pressure,
        wall_displacement,
        channel1_deformed_mesh,
        channel2_deformed_mesh,
        residual_values,
        iteration_indices,
    ) = two_dimensional_steady_fsi_dual_channel(
        fluid_solid_domain_length,
        fluid_domain_height,
        inlet_x_channel1,
        outlet_x_channel1,
        inlet_x_channel2,
        outlet_x_channel2,
        nb_mesh_nodes_x,
        nb_mesh_nodes_y,
        reynolds_number_channel1,
        reynolds_number_channel2,
        inlet_pressure_channel1,
        inlet_pressure_channel2,
        w_new,
        p_new_1,
        p_new_2,
        dimensionless_bending_stiffness,
        dimensionless_extensional_stiffness,
        under_relaxation_factor,
        residual_number,
        iteration_number,
        epsilon,
    )

    # Diagnostics
    max_disp = np.max(np.abs(wall_displacement))
    assert max_disp > 0.0, "Wall displacement should be non-zero under pressure load."

    # Optional visualization
    if plot_results:
        visualize_mesh(channel1_deformed_mesh, title="Channel 1 Deformed Mesh")
        visualize_mesh(channel2_deformed_mesh, title="Channel 2 Deformed Mesh")

        plot_dual_channel_profiles(
            wall_displacement,
            channel1_pressure,
            channel2_pressure,
            residual_values,
            iteration_indices,
            fluid_solid_domain_length,
        )


def plot_dual_channel_profiles(wall_disp, p1, p2, res, it, length):
    """
    Plots displacement, pressure profiles, and  residual vs iteration number
    """
    x = np.linspace(0, length, len(wall_disp))

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
    ax[0].plot(x, wall_disp, label="Displacement", color="green")
    ax[1].plot(x, p1, label="Channel 1 Pressure", color="blue")
    ax[2].plot(x, p2, label="Channel 2 Pressure", color="red")
    ax[3].semilogy(
        it[1:], res[1:], label="residual vs iteration number", color="orange"
    )

    ax[0].set_ylabel(r"$\tilde{w}$")
    ax[1].set_ylabel(r"$\tilde{p_1}$")
    ax[2].set_ylabel(r"$\tilde{p_2}$")
    ax[3].set_ylabel("Residual (log scale)")

    ax[0].set_xlabel(r"$\tilde{x}$")
    ax[1].set_xlabel(r"$\tilde{x}$")
    ax[2].set_xlabel(r"$\tilde{x}$")
    ax[3].set_xlabel("Cumulative Iteration Count")

    for a in ax:
        a.grid(True)
        #a.legend()

    plt.suptitle("FSI Dual Channel Results")
    plt.tight_layout()
    plt.savefig("dual channel parallel flow.png")
    plt.show()

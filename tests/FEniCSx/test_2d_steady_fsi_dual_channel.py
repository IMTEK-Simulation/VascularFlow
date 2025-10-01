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
            6,   #Re1
            6,   #Re2
            13.88,   #p1
            47.22,    #p2
            277777,   #beta
            3333333,  #gamma
            0.005, #relax
            1e-6,   #tol
            2000,    #max it
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

    #dx = abs(inlet_x_channel1 - outlet_x_channel1) / nb_mesh_nodes_x
    #dy = np.diff(wall_displacement)
    #ds = np.sqrt(dx ** 2 + dy ** 2)
    #s = np.sum(ds)
    #print(f"Arc length s = {s}")
    # Diagnostics
    #max_disp = np.max(np.abs(wall_displacement))
    #print(f"Max disp = {max_disp}")
    #assert max_disp > 0.0, "Wall displacement should be non-zero under pressure load."

    print(wall_displacement)
    print(channel1_pressure)
    print(channel2_pressure)

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
    Plots displacement, pressure profiles, and residual vs iteration number
    """
    x = np.linspace(0, length, len(wall_disp))

    # Create subplots with shared x-axis for top three plots
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(14, 11), constrained_layout=True)

    # Share x-axis for ax[1] and ax[2] with ax[0]
    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])

    # Plot data
    ax[0].plot(x, wall_disp, label="Beam displacement", color="green")
    ax[1].plot(x, p1, label="Channel 1 Pressure", color="blue")
    ax[2].plot(x, p2, label="Channel 2 Pressure", color="red")
    ax[3].semilogy(it[1:], res[1:], label="Residual vs iteration", color="orange")

    # Set y-labels (shorter and clearer, split across lines)
    ax[0].set_ylabel("Normalized displacement\n$\\tilde{w} = \\dfrac{w}{H_0}$")
    ax[1].set_ylabel("Normalized Pressure (Channel 1)\n$\\tilde{p}_1 = \\dfrac{p_{1,\\,\\mathrm{inlet}}}{\\rho U^2}$")
    ax[2].set_ylabel("Normalized Pressure (Channel 2)\n$\\tilde{p}_2 = \\dfrac{p_{2,\\,\\mathrm{inlet}}}{\\rho U^2}$")
    ax[3].set_ylabel("Residual (log scale)")

    # Set x-labels
    ax[2].set_xlabel("Normalized distance\n$\\tilde{x} = \\dfrac{x}{H_0}$")
    ax[3].set_xlabel("Cumulative Iteration Count")

    # Hide redundant x-tick labels for upper shared plots
    ax[0].tick_params(labelbottom=False)
    ax[1].tick_params(labelbottom=False)

    # Enable grid for all plots
    for a in ax:
        a.grid(True)

    # Align y-labels for top 3 plots
    fig.align_ylabels(ax[:3])

    # Set the figure title
    fig.suptitle("FSI Dual Channel (counter flow) Results")

    # Save and show
    #fig.savefig("dual channel parallel flow.png", bbox_inches="tight")
    plt.show()
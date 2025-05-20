import pytest
import numpy as np
import pyvista
from dolfinx import plot
from matplotlib import pyplot as plt


from VascularFlow.FEniCSx.Coupling.SteadyFSIDualChannel import (
    two_dimensional_steady_fsi_dual_channel,
)

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
        "iteration_number"
    ),
    [
        # Format: (length, height, inlet1, outlet1, inlet2, outlet2, nx, ny, Re1, Re2, p1, p2, B, A, relax, tol, max_iter)
        (50, 1, 0, 50, 50, 0, 500, 10, 1.0, 1.0, 100, 100, 9e4, 1.6e5, 0.00009, 1e-8, 50),
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
        visualize_dual_channel_meshes(channel1_deformed_mesh, channel2_deformed_mesh)
        plot_dual_channel_profiles(
            wall_displacement, channel1_pressure, channel2_pressure, fluid_solid_domain_length
        )


def visualize_dual_channel_meshes(mesh1, mesh2):
    """
    Visualizes deformed meshes for channel 1 and channel 2 using PyVista.
    """
    mesh1.topology.create_connectivity(2, 2)
    mesh2.topology.create_connectivity(2, 2)

    topo1, cell_types1, geom1 = plot.vtk_mesh(mesh1, 2)
    topo2, cell_types2, geom2 = plot.vtk_mesh(mesh2, 2)

    grid1 = pyvista.UnstructuredGrid(topo1, cell_types1, geom1)
    grid2 = pyvista.UnstructuredGrid(topo2, cell_types2, geom2)

    plotter1 = pyvista.Plotter()
    plotter2 = pyvista.Plotter()

    plotter1.add_mesh(grid1, show_edges=True)
    plotter2.add_mesh(grid2, show_edges=True)

    plotter1.view_xy()
    plotter2.view_xy()

    plotter1.show(title="Channel 1 Deformed Mesh")
    plotter2.show(title="Channel 2 Deformed Mesh")


def plot_dual_channel_profiles(wall_disp, p1, p2, length):
    """
    Plots displacement and pressure profiles across the channel.
    """
    x = np.linspace(0, length, len(wall_disp))

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
    ax[0].plot(x, wall_disp, label="Displacement", color="green")
    ax[1].plot(x, p1, label="Channel 1 Pressure", color="blue")
    ax[2].plot(x, p2, label="Channel 2 Pressure", color="red")

    ax[0].set_ylabel("w (displacement)")
    ax[1].set_ylabel("p1")
    ax[2].set_ylabel("p2")
    ax[2].set_xlabel("x (channel length)")

    for a in ax:
        a.grid(True)
        a.legend()

    plt.suptitle("FSI Dual Channel Results")
    plt.tight_layout()
    plt.show()

import pytest
import numpy as np
from matplotlib import pyplot as plt

from VascularFlow.FEniCSx.Coupling.SteadyFSISingleChannel import steady_state_fsi_single_channel
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh

@pytest.mark.parametrize(
    (
        "fluid_solid_domain_outlet_coordinate",
        "nb_mesh_nodes_x",
        "nb_mesh_nodes_y",
        "reynolds_number",
        "inlet_pressure",
        "dimensionless_bending_stiffness",
        "dimensionless_extensional_stiffness",
        "under_relaxation_factor",
        "residual_number",
        "iteration_number",
    ),
    [
        # Format: (L, nx, ny, Re, p_in, B, A, relax, tol, max_iter)
        (50, 500, 10, 1.0, 280.0, 9e4, 1.6e5, 0.0003, 1e-4, 2000),
    ],
)

def test_steady_state_fsi_single_channel_parametrized(
    fluid_solid_domain_outlet_coordinate,
    nb_mesh_nodes_x,
    nb_mesh_nodes_y,
    reynolds_number,
    inlet_pressure,
    dimensionless_bending_stiffness,
    dimensionless_extensional_stiffness,
    under_relaxation_factor,
    residual_number,
    iteration_number,
):
    """
    Parametrized integration test for steady-state FSI across multiple configurations.
    """

    # Fixed domain geometry and initial guesses
    fluid_solid_domain_inlet_coordinate = 0
    fluid_domain_height = 1.0

    w_new = np.zeros(nb_mesh_nodes_x + 1)
    p_new = np.ones(nb_mesh_nodes_x + 1)
    ambient_pressure = np.zeros(nb_mesh_nodes_x + 1)

    # Run FSI solver
    pressure, displacement, deformed_mesh, residual_values, iteration_indices = steady_state_fsi_single_channel(
        fluid_solid_domain_inlet_coordinate,
        fluid_solid_domain_outlet_coordinate,
        fluid_domain_height,
        nb_mesh_nodes_x,
        nb_mesh_nodes_y,
        reynolds_number,
        inlet_pressure,
        w_new,
        p_new,
        dimensionless_bending_stiffness,
        dimensionless_extensional_stiffness,
        ambient_pressure,
        under_relaxation_factor,
        residual_number,
        iteration_number,
    )

    # Diagnostics
    max_displacement = np.max(np.abs(displacement))
    max_pressure = np.max(pressure)

    #print(f"Max displacement: {max_displacement:.4f}")
    #print(f"Max pressure: {max_pressure:.4f}")

    assert max_displacement > 0.0, "Displacement should not be zero under load."
    assert max_pressure > 0.0, "Pressure must remain positive at the interface."

    # Optional visualization (toggle for local use)
    visualize = True
    if visualize:
        visualize_mesh(deformed_mesh, title="Deformed Mesh")

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        x = np.linspace(0, 50, nb_mesh_nodes_x + 1)
        # Plots
        ax[0].plot(x, pressure)
        ax[1].semilogy(iteration_indices[1:], residual_values[1:])
        # Set labels
        ax[0].set_xlabel("Normalized Channel Length (x)")
        ax[0].set_ylabel("Normalized Channel Pressure")
        ax[1].set_xlabel("Cumulative Iteration Count")
        ax[1].set_ylabel("Residual (log scale)")
        # Enable grid on both subplots
        ax[0].grid(True)
        ax[1].grid(True)
        plt.tight_layout()
        plt.savefig("single channel.png")
        plt.show()

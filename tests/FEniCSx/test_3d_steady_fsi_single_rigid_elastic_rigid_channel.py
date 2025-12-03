# import numpy as np
from matplotlib import pyplot as plt
import pytest

from VascularFlow.FEniCSx.Coupling.SteadyFsiRigidElasticRigidChannel import (
    three_dimensional_steady_fsi_single_rigid_elastic_rigid_channel,
)
from VascularFlow.Numerics.BasisFunctions import ACMShapeFunctions
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh


@pytest.mark.parametrize(
    (
        "inlet_pressure",
        "outlet_pressure",
        "plate_external_pressure_value",
        "channel_length",
        "channel_width",
        "channel_height",
        "x_max_channel_right",
        "x_min_channel_left",
    ),
    [
        (30, 0, 0, 1500/20, 200/20, 20/20, 25, 50),
    ],
)
def test_3d_steady_fsi_single_rigid_elastic_rigid_channel(
    inlet_pressure,
    outlet_pressure,
    plate_external_pressure_value,
    channel_length,
    channel_width,
    channel_height,
    x_max_channel_right,
    x_min_channel_left,
    plot_results=True,
):
    # -------------------------
    # Mesh & geometry settings
    # -------------------------
    n_x_fluid_domain = 15
    n_y_fluid_domain = 5
    n_z_fluid_domain = 5
    # -------------------------
    # Plate & fluid properties
    # -------------------------
    reynolds_number = 6
    plate_shape_function = ACMShapeFunctions()
    plate_thickness = 20e-6
    plate_young_modulus = 3e5
    plate_poisson_ratio = 0
    initial_channel_height = 20e-06
    fluid_density = 1000
    fluid_velocity = 0.3
    bc_positions = ["bottom", "right", "top", "left"]
    bc_values = [0, 0, 0, 0]
    # -------------------------
    # Solver controls
    # -------------------------
    under_relaxation_factor = 0.05
    residual_number = 1e-06
    iteration_number = 400
    epsilon = 3e-16

    # -------------------------
    # Run the FSI solver
    # -------------------------
    (
        converged_middle_top_wall_pressure,
        converged_middle_top_wall_displacement,
        deformed_mesh,
        residual_values,
        iteration_indices,
        converged_Q,
    ) = three_dimensional_steady_fsi_single_rigid_elastic_rigid_channel(
        channel_length,
        channel_width,
        channel_height,
        x_max_channel_right,
        x_min_channel_left,
        n_x_fluid_domain,
        n_y_fluid_domain,
        n_z_fluid_domain,
        inlet_pressure,
        outlet_pressure,
        reynolds_number,
        plate_shape_function,
        plate_external_pressure_value,
        plate_thickness,
        plate_young_modulus,
        plate_poisson_ratio,
        initial_channel_height,
        fluid_density,
        fluid_velocity,
        bc_positions,
        bc_values,
        under_relaxation_factor,
        residual_number,
        iteration_number,
        epsilon,
    )
    print('Q = ', converged_Q)
    print('max displacement = ', max(converged_middle_top_wall_displacement))
    visualize_mesh(deformed_mesh, title="Deformed Mesh from Interface Displacement")

    # --- Plotting ---
    if plot_results:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        ax.semilogy(
            iteration_indices[1:],
            residual_values[1:],
            label="Residual vs iteration",
            color="orange",
        )

        ax.set_title("Volumetric Flow Rate vs Pressure Drop")
        ax.set_xlabel("Residual (log scale)")
        ax.set_ylabel("Cumulative Iteration Count")
        ax.grid(True)
        ax.legend()
        plt.show()

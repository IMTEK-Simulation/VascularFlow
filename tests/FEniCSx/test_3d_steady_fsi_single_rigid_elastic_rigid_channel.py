# import numpy as np
from matplotlib import pyplot as plt
import pytest

from VascularFlow.FEniCSx.Coupling.SteadyFsiRigidElasticRigidChannel import (
    three_dimensional_steady_fsi_single_rigid_elastic_rigid_channel,
)
from VascularFlow.Numerics.BasisFunctions import ACMShapeFunctions


@pytest.mark.parametrize(
    (
        "delta_p_values",
        "plate_external_pressure_value",
        "channel_length",
        "channel_width",
        "channel_height",
        "x_max_channel_right",
        "x_min_channel_left",
    ),
    [
        ([0, 10, 20, 30, 40], [0, 1], 15, 1, 1, 5, 10),
    ],
)
def test_3d_steady_fsi_single_rigid_elastic_rigid_channel(
    delta_p_values,
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
    outlet_pressure = 0
    reynolds_number = 6
    plate_shape_function = ACMShapeFunctions()
    plate_thickness = 20e-6
    plate_young_modulus = 3e6
    plate_poisson_ratio = 0.3
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
    iteration_number = 10
    epsilon = 3e-16

    # Store results: mapping p_ext → list of Q-values across Δp
    Q_results = {p_ext: [] for p_ext in plate_external_pressure_value}

    # -------------------------
    # Run the FSI solver
    # -------------------------
    for p_ext in plate_external_pressure_value:
        for delta_p in delta_p_values:
            inlet_pressure = delta_p + outlet_pressure
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
                p_ext,
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

            Q_results[p_ext].append(converged_Q)

    # --- Plotting ---
    if plot_results:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        for p_ext in plate_external_pressure_value:
            ax.plot(
                delta_p_values,
                Q_results[p_ext],
                marker="o",
                label=f"External p = {p_ext}",
            )

        ax.set_title("Volumetric Flow Rate vs Pressure Drop")
        ax.set_xlabel(r"$\Delta p = p_{in} - p_{out}$")
        ax.set_ylabel(r"$Q$")
        ax.grid(True)
        ax.legend()
        plt.show()

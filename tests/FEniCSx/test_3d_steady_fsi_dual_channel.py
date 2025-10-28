import pytest
import numpy as np
from matplotlib import pyplot as plt


from VascularFlow.FEniCSx.Coupling.SteadyFSIDualChannel import three_dimensional_steady_fsi_dual_channel
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh
from VascularFlow.Numerics.BasisFunctions import ACMShapeFunctions

@pytest.mark.parametrize(
    (
        "fluid_domain_1_x_inlet_coordinate",
        "fluid_domain_1_x_outlet_coordinate",
        "fluid_domain_2_x_inlet_coordinate",
        "fluid_domain_2_x_outlet_coordinate",
        "n_x_fluid_domain",
        "n_y_fluid_domain",
        "n_z_fluid_domain",
        "reynolds_number_channel1",
        "reynolds_number_channel2",
        "inlet_pressure_channel1",
        "inlet_pressure_channel2",
        "plate_thickness",
        "plate_young_modulus",
        "fluid_velocity",
        "bc_positions",
        "under_relaxation_factor",
        "residual_number",
        "iteration_number",
    ),
    [
        (
            3,  # fluid_domain_1_x_inlet_coordinate
            0,  # fluid_domain_1_x_outlet_coordinate
            3,  # fluid_domain_2_x_inlet_coordinate
            0,  # fluid_domain_2_x_outlet_coordinate
            30, # n_x_fluid_domain
            10, # n_y_fluid_domain
            10, # n_z_fluid_domain
            2, # reynolds_number_channel1
            2,  # reynolds_number_channel2
            4,  # inlet_pressure_channel1
            8,  # inlet_pressure_channel2
            20e-06,  # plate_thickness
            3e06, # plate_young_modulus
            0.3,    # fluid_velocity
            ["bottom", "right", "top", "left"],  #bc_positions
            0.5,   #under_relaxation_factor
            1e-06,   #residual_number
            4,  #iteration_number
        ),
    ],
)

def test_3d_dimensional_steady_fsi_dual_channel(
    fluid_domain_1_x_inlet_coordinate,
    fluid_domain_1_x_outlet_coordinate,
    fluid_domain_2_x_inlet_coordinate,
    fluid_domain_2_x_outlet_coordinate,
    n_x_fluid_domain,
    n_y_fluid_domain,
    n_z_fluid_domain,
    reynolds_number_channel1,
    reynolds_number_channel2,
    inlet_pressure_channel1,
    inlet_pressure_channel2,
    plate_thickness,
    plate_young_modulus,
    fluid_velocity,
    bc_positions,
    under_relaxation_factor,
    residual_number,
    iteration_number,
    epsilon=3e-16,
    plot_results=True,
):
    n_x_plate = n_x_fluid_domain + 1
    n_y_plate = n_y_fluid_domain + 1
    fluid_domain_y_max_coordinate = 10
    fluid_domain_z_max_coordinate = 1
    plate_x_max_coordinate = max(fluid_domain_1_x_inlet_coordinate, fluid_domain_2_x_inlet_coordinate)
    plate_y_max_coordinate = fluid_domain_y_max_coordinate
    plate_poisson_ratio = 0.3
    initial_channel_height = 20e-06
    fluid_density = 1000
    bc_values =[0, 0, 0, 0]

    # Initial guesses
    w_new = np.zeros(n_x_plate * n_y_plate)
    p_new_1 = np.zeros((n_x_fluid_domain + 1) * (n_y_fluid_domain + 1))
    p_new_2 = np.zeros((n_x_fluid_domain + 1) * (n_y_fluid_domain + 1))
    plate_shape_function = ACMShapeFunctions()

    # Run the FSI solver
    (
        channel1_pressure,
        channel2_pressure,
        plate_displacement,
        channel1_deformed_mesh,
        channel2_deformed_mesh,
        residual_values,
        iteration_indices,
    ) = three_dimensional_steady_fsi_dual_channel(
        fluid_domain_1_x_inlet_coordinate,
        fluid_domain_1_x_outlet_coordinate,
        fluid_domain_2_x_inlet_coordinate,
        fluid_domain_2_x_outlet_coordinate,
        fluid_domain_y_max_coordinate,
        fluid_domain_z_max_coordinate,
        n_x_fluid_domain,
        n_y_fluid_domain,
        n_z_fluid_domain,
        plate_shape_function,
        plate_x_max_coordinate,
        plate_y_max_coordinate,
        n_x_plate,
        n_y_plate,
        reynolds_number_channel1,
        reynolds_number_channel2,
        inlet_pressure_channel1,
        inlet_pressure_channel2,
        plate_thickness,
        plate_young_modulus,
        plate_poisson_ratio,
        initial_channel_height,
        fluid_density,
        fluid_velocity,
        bc_positions,
        bc_values,
        w_new,
        p_new_1,
        p_new_2,
        under_relaxation_factor,
        residual_number,
        iteration_number,
        epsilon
    )

    if plot_results:

        visualize_mesh(channel1_deformed_mesh, title="Channel 1 Deformed Mesh")
        visualize_mesh(channel2_deformed_mesh, title="Channel 2 Deformed Mesh")

        # --- Extract and plot midline deflection ---
        Z = plate_displacement.reshape(n_y_plate, n_x_plate)
        mid_index_p = n_y_plate // 2
        midline_disp_p = Z[mid_index_p, :]

        x_coords = np.linspace(0, plate_x_max_coordinate, n_x_plate)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_coords, midline_disp_p)
        ax.set_title("Midline Deflection for plate along x coordinate")
        ax.set_xlabel("Normalized distance\n$\\tilde{x} = \\dfrac{x}{H_0}$")
        ax.set_ylabel("Normalized midline displacement\n$\\tilde{w} = \\dfrac{w}{H_0}$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
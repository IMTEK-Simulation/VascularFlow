import pytest
import numpy as np
from matplotlib import pyplot as plt


from VascularFlow.FEniCSx.Coupling.SteadyFSIDualChannel import (
    three_dimensional_steady_fsi_dual_channel,
)
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh
from VascularFlow.Numerics.BasisFunctions import ACMShapeFunctions


@pytest.mark.parametrize(
    (
        "fluid_domain_1_x_inlet_coordinate",
        "fluid_domain_1_x_outlet_coordinate",
        "fluid_domain_2_x_inlet_coordinate",
        "fluid_domain_2_x_outlet_coordinate",
        "reynolds_number_channel1",
        "reynolds_number_channel2",
        "inlet_pressure_channel1",
        "inlet_pressure_channel2",
        "fluid_velocity",
    ),
    [
        (50, 0, 50, 0, 6.35, 6.35, 12.4, 44.6, 0.3),
    ],
)
def test_3d_dimensional_steady_fsi_dual_channel(
    fluid_domain_1_x_inlet_coordinate,
    fluid_domain_1_x_outlet_coordinate,
    fluid_domain_2_x_inlet_coordinate,
    fluid_domain_2_x_outlet_coordinate,
    reynolds_number_channel1,
    reynolds_number_channel2,
    inlet_pressure_channel1,
    inlet_pressure_channel2,
    fluid_velocity,
    plot_results=True,
):
    # -------------------------
    # Mesh & geometry settings
    # -------------------------
    n_x_fluid_domain = 10
    n_y_fluid_domain = 10
    n_z_fluid_domain = 5
    n_x_plate = n_x_fluid_domain + 1
    n_y_plate = n_y_fluid_domain + 1
    fluid_domain_y_max_coordinate = 10
    fluid_domain_z_max_coordinate = 1
    plate_x_max_coordinate = max(
        fluid_domain_1_x_inlet_coordinate, fluid_domain_2_x_inlet_coordinate
    )
    plate_y_max_coordinate = fluid_domain_y_max_coordinate
    # -------------------------
    # Plate & fluid properties
    # -------------------------
    plate_poisson_ratio = 0.3
    plate_thickness = 20e-6
    plate_young_modulus = 3e6
    bc_positions = ["bottom", "right", "top", "left"]
    bc_values = [0, 0, 0, 0]
    initial_channel_height = 20e-06
    fluid_density = 1000
    # -------------------------
    # Solver controls
    # -------------------------
    under_relaxation_factor = 0.05
    residual_number = 1e-06
    iteration_number = 50
    epsilon = 3e-16
    # -------------------------
    # Initial guesses
    # -------------------------
    w_new = np.zeros(n_x_plate * n_y_plate)
    p_new_1 = np.zeros((n_x_fluid_domain + 1) * (n_y_fluid_domain + 1))
    p_new_2 = np.zeros((n_x_fluid_domain + 1) * (n_y_fluid_domain + 1))
    plate_shape_function = ACMShapeFunctions()

    # -------------------------
    # Run the FSI solver
    # -------------------------
    (
        channel1_pressure,
        channel2_pressure,
        plate_displacement,
        channel1_deformed_mesh,
        channel2_deformed_mesh,
        residual_values,
        iteration_indices,
        converged_Q1,
        converged_Q2,
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
        epsilon,
    )

    #print(channel2_pressure)
    visualize_mesh(channel2_deformed_mesh, title="Deformed Mesh from Interface Displacement")

    # -------------------------
    # Collect data for the final scatter plot: w_max vs Δp2
    # -------------------------
    w_max = float(np.max(plate_displacement))
    delta_p2 = float(inlet_pressure_channel2 - inlet_pressure_channel1)  # hPa
    delta_p1 = float(inlet_pressure_channel1)

    # Attach a static bucket on the test function to accumulate across param sets
    if not hasattr(test_3d_dimensional_steady_fsi_dual_channel, "w_max_data"):
        test_3d_dimensional_steady_fsi_dual_channel.w_max_data = []
    test_3d_dimensional_steady_fsi_dual_channel.w_max_data.append(
        (delta_p2, w_max, delta_p1)
    )

    if plot_results:

        fig, ax = plt.subplots(
            nrows=4, ncols=1, figsize=(5, 3), constrained_layout=True
        )
        # --- Share x-axis for ax[1] and ax[2] with ax[0] ---
        ax[1].sharex(ax[0])
        ax[2].sharex(ax[0])
        # --- Plate midline deflection ---
        Z = plate_displacement.reshape(n_y_plate, n_x_plate)
        mid_index_plate = n_y_plate // 2
        midline_disp_plate = Z[mid_index_plate, :]
        x_coordinate_deflection = np.linspace(0, plate_x_max_coordinate, n_x_plate)
        # --- Channel midline pressures (take first row) ---
        midline_pressure_channel1 = channel1_pressure[0 : n_x_fluid_domain + 1]
        midline_pressure_channel2 = channel2_pressure[0 : n_x_fluid_domain + 1]
        x_coordinate_pressure = np.linspace(
            0, plate_x_max_coordinate, n_x_fluid_domain + 1
        )

        ax[0].plot(
            x_coordinate_deflection,
            midline_disp_plate,
            label="Midline Deflection for plate along x coordinate",
            color="green",
        )
        ax[1].plot(
            x_coordinate_pressure,
            midline_pressure_channel1,
            label="Midline pressure in top channel along x coordinate",
            color="blue",
        )
        ax[2].plot(
            x_coordinate_pressure,
            midline_pressure_channel2,
            label="Midline pressure in bottom channel along x coordinate",
            color="red",
        )
        ax[3].semilogy(
            iteration_indices[1:],
            residual_values[1:],
            label="Residual vs iteration",
            color="orange",
        )

        # --- Set y-labels (shorter and clearer, split across lines) ---
        ax[0].set_ylabel("$\\tilde{w} = \\dfrac{w}{H_0}$")
        ax[1].set_ylabel("$\\tilde{p}_1 = \\dfrac{p_{1,\\,\\mathrm{inlet}}}{\\rho U^2}$")
        ax[2].set_ylabel("$\\tilde{p}_2 = \\dfrac{p_{2,\\,\\mathrm{inlet}}}{\\rho U^2}$")
        ax[3].set_ylabel("Residual (log scale)")

        # --- Set x-labels ---
        ax[2].set_xlabel("$\\tilde{x} = \\dfrac{x}{H_0}$")
        ax[3].set_xlabel("Cumulative Iteration Count")

        # --- Hide redundant x-tick labels for upper shared plots ---
        ax[0].tick_params(labelbottom=False)
        ax[1].tick_params(labelbottom=False)

        # Enable grid for all plots
        for a in ax:
            a.grid(True)

        # Align y-labels for top 3 plots
        fig.align_ylabels(ax[:3])

        # Set the figure title
        #fig.suptitle("FSI Dual Channel (counter flow) Results")

        # Save and show
        # fig.savefig("dual channel parallel flow.png", bbox_inches="tight")
        plt.show()

# ----------------------------------------------------------------------
# After ALL parameterized cases finish, make the single scatter plot:
# w_max (µm) vs Δp2 (hPa), grouped by Δp1 (hPa)
# ----------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def plot_wmax_vs_deltap2():
    """
    Session-scope autouse fixture: after all tests in this module are done,
    if we've accumulated (Δp2, w_max, Δp1) samples from the test, plot them.
    """
    # Wait until all tests complete
    yield

    data = getattr(test_3d_dimensional_steady_fsi_dual_channel, "w_max_data", [])
    if not data:
        return

    data = np.array(data, dtype=float)
    delta_p2_values = data[:, 0]            # hPa
    w_max_values_um = data[:, 1]     # convert m -> µm
    delta_p1_values = data[:, 2]            # hPa (for legend groups)

    # Group by Δp1, give each group a distinct marker/color
    unique_p1 = np.unique(delta_p1_values)
    markers = ["^", "s", "v", "o", "D", "P", "X"]
    # Use a pleasant qualitative colormap
    cmap = plt.cm.Set2(np.linspace(0, 1, len(unique_p1)))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i, p1 in enumerate(unique_p1):
        mask = delta_p1_values == p1
        ax.scatter(
            delta_p2_values[mask],
            w_max_values_um[mask],
            marker=markers[i % len(markers)],
            s=80,
            alpha=0.9,
            color=cmap[i],
            label=f"{p1:.1f}",
            edgecolors="none",
        )

    # Labels and legend to match the example style
    ax.set_xlabel(r"$\Delta p_2$ (hPa)")
    ax.set_ylabel(r"$w_{\max}$ ($\mu$m)")

    # Legend block with Δp1 header
    legend = ax.legend(
        title=r"$\Delta p_1$ (hPa)",
        frameon=True,
        loc="lower right",
    )
    legend._legend_box.align = "left"  # neat title alignment

    # Light grid & small "a)" panel label like your figure
    ax.grid(True, which="both", linestyle="-", alpha=0.3)
    ax.text(-0.12, 1.03, "a)", transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    plt.show()






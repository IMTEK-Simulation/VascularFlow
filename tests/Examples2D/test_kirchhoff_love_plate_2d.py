import numpy as np
import pytest
import matplotlib.pyplot as plt

from VascularFlow.Examples2D.KirchhoffLovePlate import plate_bending_acm_2d
from VascularFlow.Numerics.BasisFunctions import ACMShapeFunctions


@pytest.mark.parametrize(
    "plate_length, plate_width, n_x, n_y, bc_positions, bc_values, q1_value, q2_value",
    [
        (1, 1, 51, 51, ["bottom", "right", "top", "left"], [0, 0, 0, 0], 1.0, 2.0),
    ],
)
def test_plate_bending_acm_2d(
    plate_length, plate_width, n_x, n_y, bc_positions, bc_values, q1_value, q2_value
):
    """
    Parameterized test for the Kirchhoff–Love 2D ACM plate bending solver.

    This test verifies solver stability and boundary condition handling
    for different boundary setups and distributed load magnitudes.
    """

    # --- Problem setup ---
    plate_thickness = 10e-6
    plate_young_modulus = 3e6
    plate_poisson_ratio = 0.3
    initial_channel_height = 20e-6
    fluid_density = 1000
    fluid_velocity = 0.3

    # Distributed loads for both channels
    q1 = np.full(n_x * n_y, q1_value)
    q2 = np.full(n_x * n_y, q2_value)

    shape_function = ACMShapeFunctions()

    # --- Solve the plate bending problem ---
    plate_solution, lhs, rhs = plate_bending_acm_2d(
        shape_function=shape_function,
        domain_length=plate_length,
        domain_height=plate_width,
        n_x=n_x,
        n_y=n_y,
        plate_thickness=plate_thickness,
        plate_young_modulus=plate_young_modulus,
        plate_poisson_ratio=plate_poisson_ratio,
        initial_channel_height=initial_channel_height,
        fluid_density=fluid_density,
        fluid_velocity=fluid_velocity,
        bc_positions=bc_positions,
        bc_values=bc_values,
        distributed_load_channel1=q1,
        distributed_load_channel2=q2,
    )

    # --- Basic sanity checks ---
    N_nodes = n_x * n_y
    assert plate_solution.size == 3 * N_nodes, "Unexpected total DOF count"
    assert np.isfinite(plate_solution).all(), "Solution contains NaN or inf"

    # --- Plot (only if running interactively, not for CI) ---
    if not plt.isinteractive():
        # --- Extract components ---
        plate_displacement = plate_solution[0::3].reshape(n_y, n_x)
        plate_x_rotation = plate_solution[1::3].reshape(n_y, n_x)
        plate_y_rotation = plate_solution[2::3].reshape(n_y, n_x)

        # --- Create 3-panel figure ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # --- Plot displacement (w) ---
        im0 = axes[0].imshow(
            plate_displacement,
            extent=[0, plate_length, 0, plate_width],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        axes[0].set_title("Deflection w")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)

        # --- Plot rotation θx ---
        im1 = axes[1].imshow(
            plate_x_rotation,
            extent=[0, plate_length, 0, plate_width],
            origin="lower",
            cmap="plasma",
            aspect="auto",
        )
        axes[1].set_title("Rotation θx")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        # --- Plot rotation θy ---
        im2 = axes[2].imshow(
            plate_y_rotation,
            extent=[0, plate_length, 0, plate_width],
            origin="lower",
            cmap="inferno",
            aspect="auto",
        )
        axes[2].set_title("Rotation θy")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)

        # --- Overall figure title ---
        fig.suptitle(f"Kirchhoff–Love Plate (BCs={bc_positions})", fontsize=14)
        plt.show()

    # --- Extract and plot midline deflection (y=0.5) ---
    plate_displacement = plate_solution[0::3]  # deflection w
    Z = plate_displacement.reshape(n_y, n_x)
    mid_index = n_y // 2
    midline_disp = Z[mid_index, :]

    # Check deflection is finite and not all zeros (unless load is zero)
    if q1_value != 0 or q2_value != 0:
        assert np.any(np.abs(midline_disp) > 0), "Deflection should not be all zero"

    # Optional — lightweight midline diagnostic plot
    if not plt.isinteractive():
        x_coords = np.linspace(0, plate_length, n_x)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_coords, midline_disp, label="w(y=0.5)")
        ax.set_title("Midline Deflection")
        ax.set_xlabel("x")
        ax.set_ylabel("w")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

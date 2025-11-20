# -----------------------------------------------------------------------------
# 3D steady-state FSI for a single rigid–elastic–rigid channel
# -----------------------------------------------------------------------------
#
# Overview
# --------
# This routine performs a partitioned fluid–structure interaction (FSI)
# simulation for a 3D channel composed of:
#
#   - Left section: rigid walls (including top)
#   - Middle section: elastic top wall modeled as a Kirchhoff–Love plate
#   - Right section: rigid walls (including top)
#
# The coupling is done in a fixed-point (Gauss–Seidel) fashion between:
#
#   1. A 3D steady Navier–Stokes solver with pressure inlet/outlet:
#        - Solved on the current fluid mesh.
#        - Provides the pressure in the middle top wall (fluid–structure interface)
#          and the volumetric flow rate at the outlet.
#
#   2. A 2D Kirchhoff–Love plate bending solver:
#        - Uses the pressure from the fluid as external load on the plate.
#        - Returns the plate deflection (vertical displacement) at the
#          interface.
#
#   3. A mesh deformation step:
#        - Takes the interface displacement from the plate and solves a
#          Laplace problem in the fluid domain.
#        - Updates the fluid mesh coordinates (z-direction) accordingly.
#
# The coupling loop iterates until a convergence criterion is met based on
# changes in both the mesh deformation and the interface pressure:
#
#       residual = max(residual_h, residual_p)
#
# where residual_h measures relative changes in the mesh height and
# residual_p measures relative changes in the interface pressure.
#
# Output:
#   - Final converged interface pressure and deflection vectors on the
#     elastic top wall.
#   - The final deformed fluid mesh.
#   - Residual history for monitoring convergence of the FSI iterations.
# -----------------------------------------------------------------------------

import dolfinx
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.FluidFlow.Pressure2DPressureInlet import (
    pressure_3d_pressure_inlet_rigid_elastic_rigid_channel,
)
from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import (
    mesh_deformation_3d_rigid_elastic_rigid_channel,
)
from VascularFlow.Examples2D.KirchhoffLovePlate import plate_bending_acm_2d


def three_dimensional_steady_fsi_single_rigid_elastic_rigid_channel(
    channel_length: float,
    channel_width: float,
    channel_height: float,
    x_max_channel_right: float,
    x_min_channel_left: float,
    n_x_fluid_domain: int,
    n_y_fluid_domain: int,
    n_z_fluid_domain: int,
    inlet_pressure: float,
    outlet_pressure: float,
    reynolds_number: float,
    plate_shape_function,
    plate_external_pressure_value: float,
    plate_thickness: float,
    plate_young_modulus: float,
    plate_poisson_ratio: float,
    initial_channel_height: float,
    fluid_density: float,
    fluid_velocity: float,
    bc_positions,
    bc_values,
    under_relaxation_factor: float,
    residual_number: float,
    iteration_number: int,
    epsilon: float,
):
    """
    Parameters
    ----------
    channel_length : float
        Total length of the 3D channel in the x-direction.
    channel_width : float
        Width of the channel in the y-direction.
    channel_height : float
        Height of the channel in the z-direction (undeformed).
    x_max_channel_right : float
        x-coordinate of the right boundary of the right rigid channel section.
    x_min_channel_left : float
        x-coordinate of the left boundary of the left rigid channel section.
    n_x_fluid_domain : int
        Number of cells in the x-direction for the fluid mesh.
    n_y_fluid_domain : int
        Number of cells in the y-direction for the fluid mesh.
    n_z_fluid_domain : int
        Number of cells in the z-direction for the fluid mesh.
    inlet_pressure : float
        Prescribed pressure at the inlet (fluid problem).
    outlet_pressure : float
        Prescribed pressure at the outlet (fluid problem).
    reynolds_number : float
        Reynolds number for the fluid flow.
    plate_shape_function :
        Shape function or callable defining plate geometry / interpolation
        for the Kirchhoff–Love plate solver.
    plate_external_pressure_value : float
        External pressure applied on the plate in addition to the fluid load.
    plate_thickness : float
        Thickness of the plate (elastic top middle wall).
    plate_young_modulus : float
        Young's modulus of the plate material.
    plate_poisson_ratio : float
        Poisson's ratio of the plate material.
    initial_channel_height : float
        Reference channel height used in the plate model.
    fluid_density : float
        Fluid density used in the plate loading term (if needed by the plate model).
    fluid_velocity : float
        Reference fluid velocity used in the plate loading term (if needed).
    bc_positions :
        Positions for plate boundary conditions (plate model).
    bc_values :
        Values for plate boundary conditions at the specified positions.
    under_relaxation_factor : float
        Under-relaxation factor for the plate displacement update in the FSI loop
        (0 < under_relaxation_factor ≤ 1).
    residual_number : float
        Convergence tolerance for the FSI residual.
    iteration_number : int
        Maximum number of FSI iterations allowed.
    epsilon : float
        Small regularization parameter used to avoid division by zero in
        residual computations.

    Returns
    -------
    p_new : numpy.ndarray
        Converged pressure values on the top middle (elastic) wall of the channel.
    w_new : numpy.ndarray
        Converged displacement values (plate deflection) on the top middle wall.
    fluid_domain : dolfinx.mesh.Mesh
        Final deformed fluid mesh.
    residual_values : list[float]
        History of residual values over the FSI iterations.
    iteration_indices : list[int]
        Corresponding iteration indices for the residual history.
    """

    # -------------------------------------------------------------------------
    # 1. Create initial (undeformed) 3D fluid mesh
    # -------------------------------------------------------------------------

    fluid_domain = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [
            [
                0,
                0,
                0,
            ],
            [
                channel_length,
                channel_width,
                channel_height,
            ],
        ],
        [n_x_fluid_domain, n_y_fluid_domain, n_z_fluid_domain],
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )

    # -------------------------------------------------------------------------
    # 2. Determine number of points along x and y on the top middle wall
    #    (this defines the plate discretization and interface mapping)
    # -------------------------------------------------------------------------

    x_min_channel_middle = x_max_channel_right
    x_max_channel_middle = x_min_channel_left

    def wall_top_middle_channel(x):
        """Mask for vertices on the top middle (elastic) wall of the fluid domain."""
        return np.logical_and.reduce(
            (
                np.isclose(x[2], channel_height, atol=1e-8),
                x[0] >= x_min_channel_middle - 1e-8,
                x[0] <= x_max_channel_middle + 1e-8,
            )
        )

    # Locate vertices belonging to the top middle wall
    verts = dolfinx.mesh.locate_entities(fluid_domain, 0, wall_top_middle_channel)
    coords = fluid_domain.geometry.x[verts]
    # Unique x and y coordinates on the top middle wall
    n_x_middle_top_wall = np.unique(np.round(coords[:, 0], 10))
    n_y_middle_top_wall = np.unique(np.round(coords[:, 1], 10))

    # External plate pressure distribution (uniform here)
    plate_external_pressure_array = np.full(
        len(n_x_middle_top_wall) * len(n_y_middle_top_wall),
        plate_external_pressure_value,
    )

    # -------------------------------------------------------------------------
    # 3. Initialize interface displacement and pressure arrays for FSI loop
    # -------------------------------------------------------------------------
    w_new = np.zeros(len(n_x_middle_top_wall) * len(n_y_middle_top_wall))
    p_new = np.zeros(len(n_x_middle_top_wall) * len(n_y_middle_top_wall))

    # Initialize convergence tracking
    residual = 1
    iteration = 0
    residual_values = []
    iteration_indices = []

    Q_outlet_converged = None  # <-- store final flow rate

    # -------------------------------------------------------------------------
    # 4. FSI fixed-point loop
    # -------------------------------------------------------------------------
    while residual > residual_number and iteration < iteration_number:
        # -------------------------------------------------------------
        # 4.1 Fluid solve on current fluid mesh
        #     Get interface pressure in the middle top wall, and outlet flow
        # -------------------------------------------------------------

        (
            mixed_function,
            Q_outlet,
            top_middle_wall_pressure,
            channel_length_middle,
            n_x_top_middle_wall,
            n_y_top_middle_wall,
        ) = pressure_3d_pressure_inlet_rigid_elastic_rigid_channel(
            fluid_domain,
            channel_length,
            channel_width,
            channel_height,
            x_max_channel_right,
            x_min_channel_left,
            inlet_pressure,
            outlet_pressure,
            reynolds_number,
        )

        Q_outlet_converged = Q_outlet  # <-- capture the most recent one

        # Plate dimensions based on middle section and channel width
        plate_x_max_coordinate = channel_length_middle
        plate_y_max_coordinate = channel_width
        n_x_plate = len(n_x_top_middle_wall)
        n_y_plate = len(n_y_top_middle_wall)

        # -------------------------------------------------------------
        # 4.2 Plate bending solve: get updated interface displacement
        # -------------------------------------------------------------

        plate_solution, lhs, rhs = plate_bending_acm_2d(
            plate_shape_function,
            plate_x_max_coordinate,
            plate_y_max_coordinate,
            n_x_plate,
            n_y_plate,
            plate_thickness,
            plate_young_modulus,
            plate_poisson_ratio,
            initial_channel_height,
            fluid_density,
            fluid_velocity,
            bc_positions,
            bc_values,
            plate_external_pressure_array,
            top_middle_wall_pressure,
        )
        # Extract vertical displacement component from plate solution
        w_star = plate_solution[0::3]
        # Filter out tiny numerical noise
        w_star[np.abs(w_star) < 1e-7] = 0
        # Under-relax the displacement to stabilize coupling iterations
        w_star = (
            under_relaxation_factor * w_star + (1 - under_relaxation_factor) * w_new
        )
        # -------------------------------------------------------------
        # 4.3 Recreate undeformed fluid mesh
        #     (reset before applying new deformation)
        # -------------------------------------------------------------
        fluid_domain = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    channel_length,
                    channel_width,
                    channel_height,
                ],
            ],
            [n_x_fluid_domain, n_y_fluid_domain, n_z_fluid_domain],
            cell_type=dolfinx.mesh.CellType.hexahedron,
        )
        # -------------------------------------------------------------
        # 4.4 Mesh deformation: extend plate displacement into fluid mesh
        # -------------------------------------------------------------
        fluid_domain_star = mesh_deformation_3d_rigid_elastic_rigid_channel(
            w_star,
            fluid_domain,
            channel_length,
            channel_width,
            channel_height,
            x_min_channel_left,
            x_max_channel_right,
        )

        # -------------------------------------------------------------
        # 4.5 Compute residuals for mesh height and pressure
        # -------------------------------------------------------------
        # Residual based on mesh height changes in z-direction
        if max(abs(fluid_domain.geometry.x[:, 2] - channel_height)) < epsilon:
            residual_h = max(
                abs(fluid_domain_star.geometry.x[:, 2] - fluid_domain.geometry.x[:, 2])
            ) / (max(abs(fluid_domain.geometry.x[:, 2])) + epsilon)
        else:
            residual_h = max(
                abs(fluid_domain_star.geometry.x[:, 2] - fluid_domain.geometry.x[:, 2])
            ) / max(abs(fluid_domain.geometry.x[:, 2]))
        # Residual based on changes in interface pressure
        if max(abs(p_new)) < epsilon:
            residual_p = max(abs(top_middle_wall_pressure - p_new)) / (
                max(abs(p_new)) + epsilon
            )
        else:
            residual_p = max(abs(top_middle_wall_pressure - p_new)) / max(abs(p_new))
        # Overall FSI residual
        residual = max(residual_h, residual_p)

        # -------------------------------------------------------------
        # 4.6 Update state for next iteration
        # -------------------------------------------------------------

        p_new = top_middle_wall_pressure
        w_new = w_star
        fluid_domain = fluid_domain

        iteration += 1
        residual_values.append(residual)
        iteration_indices.append(iteration)

    # -------------------------------------------------------------------------
    # 5. Return converged quantities and residual history
    # -------------------------------------------------------------------------
    return (
        p_new,
        w_new,
        fluid_domain,
        residual_values,
        iteration_indices,
        Q_outlet_converged,
    )

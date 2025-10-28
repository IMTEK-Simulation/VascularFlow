# -----------------------------------------------------------------------------
# Steady-State 2D Fluid–Structure Interaction (FSI): Dual-Channel Configuration
# -----------------------------------------------------------------------------
# This function solves a coupled steady-state fluid–structure interaction (FSI)
# problem involving two channels : parallel flow and counter flow
# separated by a common elastic interface modeled as a 1D Euler–Bernoulli beam.
# - Both channels independently experience pressure-driven flow.
# - The pressure difference from the two channels is applied as load on the beam.
# - The wall deformation is used to update the fluid domain meshes via harmonic extension.
#
# The problem is solved iteratively using a fixed-point (Picard) approach until
# displacement and pressure changes between iterations fall below a given tolerance.


import dolfinx
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.FluidFlow.Pressure2DPressureInlet import (
    pressure_2d_pressure_inlet,
)
from VascularFlow.FEniCSx.Elasticity.Beam import euler_bernoulli_steady_fenicsx
from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import mesh_deformation


def two_dimensional_steady_fsi_dual_channel(
    fluid_solid_domain_length: int,
    fluid_domain_height: int,
    inlet_x_channel1: int,
    outlet_x_channel1: int,
    inlet_x_channel2: int,
    outlet_x_channel2: int,
    nb_mesh_nodes_x: int,
    nb_mesh_nodes_y: int,
    reynolds_number_channel1: float,
    reynolds_number_channel2: float,
    inlet_pressure_channel1: float,
    inlet_pressure_channel2: float,
    w_new: np.array,
    p_new_1: np.array,
    p_new_2: np.array,
    dimensionless_bending_stiffness: float,
    dimensionless_extensional_stiffness: float,
    under_relaxation_factor: float,
    residual_number: float,
    iteration_number: int,
    epsilon: float,
):
    """
    Solves a steady-state 2D FSI problem for a dual-channel system with an elastic wall.

    Parameters
    ----------
    fluid_solid_domain_length : float
        Length of the fluid-structure domain (x-direction).

    fluid_domain_height : float
        Height of each fluid channel (y-direction).

    inlet_x_channel1 : float
        X-coordinate of the inlet for channel 1 (top channel).

    outlet_x_channel1 : float
        X-coordinate of the outlet for channel 1.

    inlet_x_channel2 : float
        X-coordinate of the inlet for channel 2 (bottom channel).

    outlet_x_channel2 : float
        X-coordinate of the outlet for channel 2.

    nb_mesh_nodes_x : int
        Number of mesh divisions along the x-direction.

    nb_mesh_nodes_y : int
        Number of mesh divisions along the y-direction.

    reynolds_number_channel1 : float
        Reynolds number for channel 1 flow.

    reynolds_number_channel2 : float
        Reynolds number for channel 2 flow.

    inlet_pressure_channel1 : float
        Inlet pressure for channel 1.

    inlet_pressure_channel2 : float
        Inlet pressure for channel 2.

    w_new : np.ndarray
        Initial displacement guess for the elastic interface.

    p_new_1 : np.ndarray
        Initial pressure guess along channel 1.

    p_new_2 : np.ndarray
        Initial pressure guess along channel 2.

    dimensionless_bending_stiffness : float
        Bending stiffness of the elastic wall (dimensionless).

    dimensionless_extensional_stiffness : float
        Extensional stiffness of the elastic wall (dimensionless).

    under_relaxation_factor : float
        Relaxation factor for stabilizing iterative updates.

    residual_number : float
        Convergence tolerance for pressure/displacement residual.

    iteration_number : int
        Maximum number of allowed iterations.

    epsilon : float
        Small number to avoid division by zero in residual calculations.

    Returns
    -------
    p_new_1 : np.ndarray
        Final converged pressure along the channel 1.

    p_new_2 : np.ndarray
        Final converged pressure along the channel 2 .

    w_new : np.ndarray
        Final converged displacement of the elastic wall.

    channel1_domain : dolfinx.mesh.Mesh
        Final deformed mesh for channel 1.

    channel2_domain : dolfinx.mesh.Mesh
        Final deformed mesh for channel 2.
    """

    # Create initial fluid meshes for channel 1 and channel 2 (rectangular domains)
    channel1_domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [fluid_solid_domain_length, fluid_domain_height]]),
        [nb_mesh_nodes_x, nb_mesh_nodes_y],
        cell_type=dolfinx.mesh.CellType.triangle,
    )

    channel2_domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [fluid_solid_domain_length, fluid_domain_height]]),
        [nb_mesh_nodes_x, nb_mesh_nodes_y],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    # Create 1D solid domain (beam interface) shared between the two channels
    solid_domain = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, nb_mesh_nodes_x, [0, fluid_solid_domain_length]
    )
    # Initialize residual and iteration count
    residual = 1
    iteration = 0

    # Store required variables at each iteration for making residual vs iteration number plot
    residual_values = []
    iteration_indices = []

    # Begin fixed-point FSI iteration loop
    while residual > residual_number and iteration < iteration_number:

        # --- Step 1: Solve fluid pressure in channel 1 ---
        mixed_function1, p1 = pressure_2d_pressure_inlet(
            channel1_domain,
            inlet_x_channel1,
            outlet_x_channel1,
            reynolds_number_channel1,
            inlet_pressure_channel1,
            navier_stokes=False,
        )

        # --- Step 2: Solve fluid pressure in channel 2 ---
        mixed_function2, p2 = pressure_2d_pressure_inlet(
            channel2_domain,
            inlet_x_channel2,
            outlet_x_channel2,
            reynolds_number_channel2,
            inlet_pressure_channel2,
            navier_stokes=False,
        )

        # --- Step 3: Solve for wall displacement due to differential pressure ---
        w_star = euler_bernoulli_steady_fenicsx(
            solid_domain,
            fluid_solid_domain_length,
            dimensionless_bending_stiffness,
            dimensionless_extensional_stiffness,
            p2,
            p1,
            "large_deflection",
        )
        # Apply under-relaxation to stabilize updates
        w_star = (
            under_relaxation_factor * w_star + (1 - under_relaxation_factor) * w_new
        )


        # --- Step 4: Update fluid meshes using harmonic extension ---
        channel1_domain = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [fluid_solid_domain_length, fluid_domain_height]]),
            [nb_mesh_nodes_x, nb_mesh_nodes_y],
            cell_type=dolfinx.mesh.CellType.triangle,
        )
        channel1_domain_star = mesh_deformation(
            -np.array(w_star), # Negative displacement for channel 1
            50,
            channel1_domain,
            harmonic_extension=True,
        )

        channel2_domain = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [fluid_solid_domain_length, fluid_domain_height]]),
            [nb_mesh_nodes_x, nb_mesh_nodes_y],
            cell_type=dolfinx.mesh.CellType.triangle,
        )
        channel2_domain_star = mesh_deformation(
            np.array(w_star), # Positive displacement for channel 2
            50,
            channel2_domain,
            harmonic_extension=True,
        )

        # --- Step 5: Compute relative residuals for convergence ---
        if max(abs(channel1_domain.geometry.x[:, 1] - 1)) < epsilon:
            residual_h_channel1 = max(
                abs(
                    channel1_domain_star.geometry.x[:, 1]
                    - channel1_domain.geometry.x[:, 1]
                )
            ) / (max(abs(channel1_domain.geometry.x[:, 1])) + epsilon)
        else:
            residual_h_channel1 = max(
                abs(
                    channel1_domain_star.geometry.x[:, 1]
                    - channel1_domain.geometry.x[:, 1]
                )
            ) / max(abs(channel1_domain.geometry.x[:, 1]))

        if max(abs(channel2_domain.geometry.x[:, 1] - 1)) < epsilon:
            residual_h_channel2 = max(
                abs(
                    channel2_domain_star.geometry.x[:, 1]
                    - channel2_domain.geometry.x[:, 1]
                )
            ) / (max(abs(channel2_domain.geometry.x[:, 1])) + epsilon)
        else:
            residual_h_channel2 = max(
                abs(
                    channel2_domain_star.geometry.x[:, 1]
                    - channel2_domain.geometry.x[:, 1]
                )
            ) / max(abs(channel2_domain.geometry.x[:, 1]))

        if max(abs(p_new_1)) < epsilon:
            residual_p_channel1 = max(abs(p1 - p_new_1)) / (max(abs(p_new_1)) + epsilon)
        else:
            residual_p_channel1 = max(abs(p1 - p_new_1)) / max(abs(p_new_1))

        if max(abs(p_new_2)) < epsilon:
            residual_p_channel2 = max(abs(p2 - p_new_2)) / (max(abs(p_new_2)) + epsilon)
        else:
            residual_p_channel2 = max(abs(p2 - p_new_2)) / max(abs(p_new_2))

        # Combine pressure and mesh residuals
        residual_channel1 = max(residual_h_channel1, residual_p_channel1)
        residual_channel2 = max(residual_h_channel2, residual_p_channel2)
        residual = max(residual_channel1, residual_channel2)

        # --- Step 6: Update state for next iteration ---
        p_new_1 = p1
        p_new_2 = p2
        w_new = w_star
        channel1_domain = channel1_domain_star
        channel2_domain = channel2_domain_star

        iteration += 1
        #print('iteration = ',iteration)

        residual_values.append(residual)
        iteration_indices.append(iteration)

    # Final return of converged pressure fields, wall shape, and deformed domains
    return p_new_1, p_new_2, w_new, channel1_domain, channel2_domain, residual_values, iteration_indices

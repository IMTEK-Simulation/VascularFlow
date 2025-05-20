# -----------------------------------------------------------------------------
# Steady-State Fluid–Structure Interaction (FSI) Solver: Single micro-channel
# -----------------------------------------------------------------------------
# This function solves a steady-state 2D fluid–structure interaction (FSI) problem
# involving a viscous incompressible fluid flowing through a micro-channel bounded
# by a flexible elastic top wall modeled using the Euler–Bernoulli beam theory.
#
# The FSI solution is obtained using a fixed-point iterative scheme that couples:
#   1. A Navier-Stokes (or Stokes) fluid solver with a pressure inlet boundary condition.
#   2. A 1D Euler–Bernoulli beam solver for the top wall deformation.
#   3. A mesh deformation method that maps interface displacements into
#      the fluid mesh using harmonic extension.
#
# The solution iterates until the displacement and pressure residuals
# fall below a given tolerance or a maximum number of iterations is reached.


import dolfinx
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.FluidFlow.Pressure2DPressureInlet import (
    pressure_2d_pressure_inlet,
)
from VascularFlow.FEniCSx.Elasticity.Beam import euler_bernoulli_steady_fenicsx
from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import mesh_deformation


def steady_state_fsi_single_channel(
    fluid_solid_domain_inlet_coordinate: int,
    fluid_solid_domain_outlet_coordinate: int,
    fluid_domain_height: float,
    nb_mesh_nodes_x: int,
    nb_mesh_nodes_y: int,
    reynolds_number: float,
    inlet_pressure: float,
    w_new: np.array,
    p_new: np.array,
    dimensionless_bending_stiffness: float,
    dimensionless_extensional_stiffness: float,
    ambient_pressure: np.ndarray,
    under_relaxation_factor: float,
    residual_number: float,
    iteration_number: int,
):
    """
    Solves the steady-state fluid–structure interaction (FSI) problem for a single
    fluid channel with an elastic top wall using a fixed-point iteration approach.

    Parameters
    ----------
    fluid_solid_domain_inlet_coordinate : int
        The x-coordinate of the fluid (solid) domain inlet.

    fluid_solid_domain_outlet_coordinate : int
        The x-coordinate of the fluid (solid) domain outlet.

    fluid_domain_height : float
        The height of the fluid domain (distance from bottom to top wall).

    nb_mesh_nodes_x : int
        Number of mesh nodes along the x-axis.

    nb_mesh_nodes_y : int
        Number of mesh nodes along the y-axis.

    reynolds_number : float
        The Reynolds number used to characterize the fluid flow (used in the Navier-Stokes equations).

    inlet_pressure : float
        The imposed pressure at the fluid inlet boundary.

    w_new : np.array
        Initial guess for the wall displacement along the interface (top wall).

    p_new : np.array
        Initial guess for the pressure profile along the interface.

    dimensionless_bending_stiffness : float
        Dimensionless bending stiffness of the elastic top wall.

    dimensionless_extensional_stiffness : float
        Dimensionless extensional stiffness (axial resistance) of the elastic top wall.

    ambient_pressure : np.array
        External load on top of the elastic top wall.

    under_relaxation_factor : float
        Factor (0 < α ≤ 1) used to stabilize the coupling iteration by blending new and previous wall displacements.

    residual_number : float
        Convergence threshold for the FSI residual (displacement and pressure).

    iteration_number : int
        Maximum number of fixed-point iterations allowed.

    Returns
    -------
    p_new : np.array
        Final converged fluid pressure along the channel.

    w_new : np.array
        Final converged wall displacement profile along the elastic interface.

    fluid_domain : dolfinx.mesh.Mesh
        Final deformed fluid domain mesh after convergence.
    """

    # Create the initial undeformed fluid domain mesh (2D rectangular domain)
    fluid_domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array(
            [
                [
                    fluid_solid_domain_inlet_coordinate,
                    fluid_solid_domain_inlet_coordinate,
                ],
                [fluid_solid_domain_outlet_coordinate, fluid_domain_height],
            ]
        ),
        [nb_mesh_nodes_x, nb_mesh_nodes_y],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    # Create the solid domain as a 1D interval mesh for the beam (top wall)
    solid_domain = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, nb_mesh_nodes_x, [0, fluid_solid_domain_outlet_coordinate]
    )
    # Initialize residual and iteration counter
    residual = 1
    iteration = 0
    # Begin fixed-point iteration loop for FSI coupling
    while residual > residual_number and iteration < iteration_number:
        # ----------------------------------------
        # Step 1: Solve the fluid flow problem (Stokes or Navier–Stokes)
        # ----------------------------------------
        mixed_function, p = pressure_2d_pressure_inlet(
            fluid_domain,
            fluid_solid_domain_inlet_coordinate,
            fluid_solid_domain_outlet_coordinate,
            reynolds_number,
            inlet_pressure,
            navier_stokes=False, # Set to True to include convection
        )

        # ----------------------------------------
        # Step 2: Solve the solid mechanics problem (Euler–Bernoulli beam)
        # ----------------------------------------
        w_star = euler_bernoulli_steady_fenicsx(
            solid_domain,
            fluid_solid_domain_outlet_coordinate,
            dimensionless_bending_stiffness,
            dimensionless_extensional_stiffness,
            p, # Pressure from fluid applied to the wall
            ambient_pressure,
            linera=True,
        )
        # Under-relaxation for numerical stability of the coupled system
        w_star = (
            under_relaxation_factor * w_star + (1 - under_relaxation_factor) * w_new
        )

        # ----------------------------------------
        # Step 3: Apply mesh deformation to fluid domain based on wall motion
        # ----------------------------------------
        fluid_domain_star = mesh_deformation(
            w_star,
            fluid_solid_domain_outlet_coordinate,
            fluid_domain,
            harmonic_extension=True,
        )

        # ----------------------------------------
        # Step 4: Compute convergence residuals
        # ----------------------------------------
        # Displacement residual: max relative change in mesh y-coordinate
        residual_h = np.max(
            np.abs(fluid_domain_star.geometry.x[:, 1] - fluid_domain.geometry.x[:, 1])
        ) / np.max(np.abs(fluid_domain.geometry.x[:, 1]))
        # Pressure residual: max relative change in interface pressure
        residual_p = np.max(np.abs(p - p_new)) / np.max(np.abs(p_new))
        # Take the worst-case residual
        residual = max(residual_h, residual_p)
        print(residual)
        # ----------------------------------------
        # Step 5: Update variables for next iteration
        # ----------------------------------------
        p_new = p
        w_new = w_star
        fluid_domain = fluid_domain_star

        iteration += 1
        print("iteration number is :", iteration)
    # Return the final pressure, wall displacement, and deformed fluid mesh
    return p_new, w_new, fluid_domain

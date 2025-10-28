# -----------------------------------------------------------------------------
# Steady-State 2D/3D Fluid–Structure Interaction (FSI): Dual-Channel Configuration
# -----------------------------------------------------------------------------
# This function solves a coupled steady-state fluid–structure interaction (FSI)
# problem involving two channels : parallel flow and counter flow
# separated by a common elastic interface modeled as:
#   - a 1D Euler–Bernoulli beam for 2D FSI model.
#   - a 2D Kirchhoff–Love plate for 3D FSI model.
#
# - Both channels independently experience pressure-driven flow.
# - The pressure difference from the two channels is applied as load on the beam or plate.
# - the deformation of the solid part is used to update the fluid domain meshes via harmonic extension.
#
# The problem is solved iteratively using a fixed-point (Picard) approach until
# displacement and pressure changes between iterations fall below a given tolerance.


import dolfinx
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.FluidFlow.Pressure2DPressureInlet import (
    pressure_2d_pressure_inlet,
    pressure_3d_pressure_inlet,
)
from VascularFlow.FEniCSx.Elasticity.Beam import euler_bernoulli_steady_fenicsx


from VascularFlow.Examples2D.KirchhoffLovePlate import plate_bending_acm_2d
from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import (
    mesh_deformation,
    mesh_deformation_3d,
)


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
            -np.array(w_star),  # Negative displacement for channel 1
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
            np.array(w_star),  # Positive displacement for channel 2
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
        # print('iteration = ',iteration)

        residual_values.append(residual)
        iteration_indices.append(iteration)

    # Final return of converged pressure fields, wall shape, and deformed domains
    return (
        p_new_1,
        p_new_2,
        w_new,
        channel1_domain,
        channel2_domain,
        residual_values,
        iteration_indices,
    )


def three_dimensional_steady_fsi_dual_channel(
    fluid_domain_1_x_inlet_coordinate: float,
    fluid_domain_1_x_outlet_coordinate: float,
    fluid_domain_2_x_inlet_coordinate: float,
    fluid_domain_2_x_outlet_coordinate: float,
    fluid_domain_y_max_coordinate: float,
    fluid_domain_z_max_coordinate: float,
    n_x_fluid_domain: int,
    n_y_fluid_domain: int,
    n_z_fluid_domain: int,
    plate_shape_function,
    plate_x_max_coordinate: float,
    plate_y_max_coordinate: float,
    n_x_plate: int,
    n_y_plate: int,
    reynolds_number_channel1: float,
    reynolds_number_channel2: float,
    inlet_pressure_channel1: float,
    inlet_pressure_channel2: float,
    plate_thickness: float,
    plate_young_modulus: float,
    plate_poisson_ratio: float,
    initial_channel_height: float,
    fluid_density: float,
    fluid_velocity: float,
    bc_positions,
    bc_values,
    w_new: np.array,
    p_new_1: np.array,
    p_new_2: np.array,
    under_relaxation_factor: float,
    residual_number: float,
    iteration_number: int,
    epsilon: float,
):
    """
    Parameters
    ----------
    fluid_domain_1_x_inlet_coordinate: float
        The x-coordinate specifying the inlet position of the top channel (channel_1).
    fluid_domain_1_x_outlet_coordinate: float
        The x-coordinate specifying the outlet position of the top channel (channel_1).
    fluid_domain_2_x_inlet_coordinate: float
        The x-coordinate specifying the inlet position of the bottom channel (channel_2).
    fluid_domain_2_x_outlet_coordinate: float
        The x-coordinate specifying the outlet position of the bottom channel (channel_2).
    fluid_domain_y_max_coordinate: float
        The y-coordinate specifying the maximum width of the both channels.
    fluid_domain_z_max_coordinate: float
        The z-coordinate specifying the maximum height of the both channels.
    n_x_fluid_domain: int
        Number of mesh divisions for the fluid domains along the x-direction.
    n_y_fluid_domain: int
        Number of mesh divisions for the fluid domains along the y-direction.
    n_z_fluid_domain: int
        Number of mesh divisions for the fluid domains along the z-direction.
    plate_shape_function: object
        The shape function instance compatible with the ACM element used for solid (plate) solver.
    plate_x_max_coordinate: float
        The x-coordinate specifying the maximum length of the plate.
    plate_y_max_coordinate: float
        The y-coordinate specifying the maximum width of the plate.
    n_x_plate: int
        Number of mesh divisions for the plate along the x-direction.
    n_y_plate: int
        Number of mesh divisions for the plate along the y-direction.
    reynolds_number_channel1: float
        Reynolds number for the top channel (channel_1) flow.
    reynolds_number_channel2: float
        Reynolds number for the bottom channel (channel_2) flow.
    inlet_pressure_channel1: float
        inlet pressure for the top channel (channel_1). The outlet pressure is zero.
    inlet_pressure_channel2: float
        outlet pressure for the bottom channel (channel_2). The outlet pressure is zero.
    plate_thickness : float
        Plate thickness t. used to calculate the non-dimensional flexural rigidity of the plate (D).
    plate_young_modulus : float
        Young’s modulus E of the plate. used to calculate the non-dimensional flexural rigidity of the plate (D).
    plate_poisson_ratio : float
        Poisson’s ratio ν of the plate. used to calculate the non-dimensional flexural rigidity of the plate (D).
    initial_channel_height : float
        Reference height H0 used in the non-dimensionalization (appears in D).
    fluid_density : float
        Fluid density ρ used in the non-dimensionalization (appears in D).
    fluid_velocity : float
        Fluid velocity U used in the non-dimensionalization (appears in D).
    bc_positions : list[str] or tuple[str, ...]
        Boundary position specifiers understood by `boundary_dofs_acm_2d`
        (e.g., ["bottom", "right", "top", "left"]).
    bc_values : list[float]
        Dirichlet values (one per boundary group). Can also be extended to
        per-DOF values if your BC routine supports it.
    w_new : np.ndarray
        Initial displacement guess for the elastic interface.
    p_new_1 : np.ndarray
        Initial pressure guess for the top channel (channel_1).
    p_new_2 : np.ndarray
        Initial pressure guess for the bottom channel (channel_2).
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
    kirchhoff_Love_plate_solution : np.ndarray, shape (3 * n_x * n_y,)
        Global solution vector ordered by node with 3 DOFs per node: [w, θx, θy].
    p_new_1 : np.ndarray
        Final converged pressure along the top channel (channel_1).
    p_new_2 : np.ndarray
        Final converged pressure along the bottom channel (channel_2).
    channel1_domain : dolfinx.mesh.Mesh
        Final deformed mesh for the top channel (channel_1).
    channel2_domain : dolfinx.mesh.Mesh
        Final deformed mesh for the bottom channel (channel_1).
    """
    # --- Build initial hexahedral meshes for both channels (unit cube) ---
    channel1_domain = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        n_x_fluid_domain,
        n_y_fluid_domain,
        n_z_fluid_domain,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )

    # Scale channel 1 to its physical extents (x length is inlet→outlet span)
    channel1_domain.geometry.x[:, 0] *= max(
        fluid_domain_1_x_inlet_coordinate, fluid_domain_1_x_outlet_coordinate
    )
    channel1_domain.geometry.x[:, 1] *= fluid_domain_y_max_coordinate
    channel1_domain.geometry.x[:, 2] *= fluid_domain_z_max_coordinate

    channel2_domain = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        n_x_fluid_domain,
        n_y_fluid_domain,
        n_z_fluid_domain,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )

    # Scale channel 2 to its physical extents
    channel2_domain.geometry.x[:, 0] *= max(
        fluid_domain_2_x_inlet_coordinate, fluid_domain_2_x_outlet_coordinate
    )
    channel2_domain.geometry.x[:, 1] *= fluid_domain_y_max_coordinate
    channel2_domain.geometry.x[:, 2] *= fluid_domain_z_max_coordinate
    # Initialize convergence tracking
    residual = 1
    iteration = 0
    residual_values = []
    iteration_indices = []
    # --- Fixed-point / Gauss–Seidel style FSI loop ---
    while residual > residual_number and iteration < iteration_number:
        # 1) Solve fluid in channel 1 with given inlet pressure and Re.
        #    Returns the mixed solution and the pressure sampled/sorted on the top face.
        mixed_function1, p1 = pressure_3d_pressure_inlet(
            fluid_domain_1_x_inlet_coordinate,
            fluid_domain_1_x_outlet_coordinate,
            fluid_domain_y_max_coordinate,
            fluid_domain_z_max_coordinate,
            n_x_fluid_domain,
            n_y_fluid_domain,
            n_z_fluid_domain,
            reynolds_number_channel1,
            inlet_pressure_channel1,
        )

        # 2) Solve fluid in channel 2.
        mixed_function2, p2 = pressure_3d_pressure_inlet(
            fluid_domain_2_x_inlet_coordinate,
            fluid_domain_2_x_outlet_coordinate,
            fluid_domain_y_max_coordinate,
            fluid_domain_z_max_coordinate,
            n_x_fluid_domain,
            n_y_fluid_domain,
            n_z_fluid_domain,
            reynolds_number_channel2,
            inlet_pressure_channel2,
        )

        # 3) Plate bending due to differential pressure (p1 on one side, p2 on the other).
        #    The plate solver returns the global vector with 3 DOFs per node: [w, θx, θy].

        # shape_function = ACMShapeFunctions()
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
            p1,
            p2,
        )
        # Extract transverse displacement w (filter tiny numerical noise)
        w_star = plate_solution[0::3]
        w_star[np.abs(w_star) < 1e-7] = 0

        # Under-relax the displacement to stabilize coupling iterations
        w_star = (
            under_relaxation_factor * w_star + (1 - under_relaxation_factor) * w_new
        )

        # 4) Update fluid meshes by solving Laplace problems with Dirichlet BC on z=Lz.
        #    Top channel moves opposite to plate deflection on the interface; bottom follows plate.
        fluid_domain_x_max_coordinate = max(
            fluid_domain_1_x_inlet_coordinate, fluid_domain_1_x_outlet_coordinate
        )

        channel1_domain_star = mesh_deformation_3d(
            -w_star,  # interface displacement sign convention for channel 1
            fluid_domain_x_max_coordinate,
            fluid_domain_y_max_coordinate,
            fluid_domain_z_max_coordinate,
            n_x_fluid_domain,
            n_y_fluid_domain,
            n_z_fluid_domain,
        )

        channel2_domain_star = mesh_deformation_3d(
            w_star,  # channel 2 moves with the plate
            fluid_domain_x_max_coordinate,
            fluid_domain_y_max_coordinate,
            fluid_domain_z_max_coordinate,
            n_x_fluid_domain,
            n_y_fluid_domain,
            n_z_fluid_domain,
        )

        # 5) Compute relative residuals for mesh motion and pressure to test convergence.
        #    Use epsilon safeguards to avoid division by ~0 during early iterations.
        if (
            max(abs(channel1_domain.geometry.x[:, 2] - fluid_domain_z_max_coordinate))
            < epsilon
        ):
            residual_h_channel1 = max(
                abs(
                    channel1_domain_star.geometry.x[:, 2]
                    - channel1_domain.geometry.x[:, 2]
                )
            ) / (max(abs(channel1_domain.geometry.x[:, 2])) + epsilon)
        else:
            residual_h_channel1 = max(
                abs(
                    channel1_domain_star.geometry.x[:, 2]
                    - channel1_domain.geometry.x[:, 2]
                )
            ) / max(abs(channel1_domain.geometry.x[:, 2]))

        if (
            max(abs(channel2_domain.geometry.x[:, 2] - fluid_domain_z_max_coordinate))
            < epsilon
        ):
            residual_h_channel2 = max(
                abs(
                    channel2_domain_star.geometry.x[:, 2]
                    - channel2_domain.geometry.x[:, 2]
                )
            ) / (max(abs(channel2_domain.geometry.x[:, 2])) + epsilon)
        else:
            residual_h_channel2 = max(
                abs(
                    channel2_domain_star.geometry.x[:, 2]
                    - channel2_domain.geometry.x[:, 2]
                )
            ) / max(abs(channel2_domain.geometry.x[:, 2]))

        # Pressure residuals (relative change w.r.t. previous iterate on each channel)
        if max(abs(p_new_1)) < epsilon:
            residual_p_channel1 = max(abs(p1 - p_new_1)) / (max(abs(p_new_1)) + epsilon)
        else:
            residual_p_channel1 = max(abs(p1 - p_new_1)) / max(abs(p_new_1))

        if max(abs(p_new_2)) < epsilon:
            residual_p_channel2 = max(abs(p2 - p_new_2)) / (max(abs(p_new_2)) + epsilon)
        else:
            residual_p_channel2 = max(abs(p2 - p_new_2)) / max(abs(p_new_2))

        # Overall residual is the max of mesh and pressure residuals across both channels
        residual_channel1 = max(residual_h_channel1, residual_p_channel1)
        residual_channel2 = max(residual_h_channel2, residual_p_channel2)
        residual = max(residual_channel1, residual_channel2)

        # 6) Accept this iterate: update state for the next coupling iteration
        p_new_1 = p1
        p_new_2 = p2
        w_new = w_star
        channel1_domain = channel1_domain_star
        channel2_domain = channel2_domain_star

        iteration += 1
        # print('iteration = ',iteration)

        residual_values.append(residual)
        iteration_indices.append(iteration)
    # Return converged pressures, displacement, final meshes, and residual history
    return (
        p_new_1,
        p_new_2,
        w_new,
        channel1_domain,
        channel2_domain,
        residual_values,
        iteration_indices,
    )

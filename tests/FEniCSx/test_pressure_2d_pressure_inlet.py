import pytest
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
import dolfinx

from VascularFlow.FEniCSx.FluidFlow.Pressure2DPressureInlet import (
    pressure_2d_pressure_inlet,
    pressure_3d_pressure_inlet,
)
from VascularFlow.FEniCSx.PostProcessing.VisualizdMixed import visualize_mixed
from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import mesh_deformation


@pytest.mark.parametrize(
    "fluid_domain, inlet_coordinate, outlet_coordinate, reynolds_number, inlet_pressure",
    [
        (
            dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD,
                np.array([[0, 0], [50, 1]]),
                [500, 10],
                cell_type=dolfinx.mesh.CellType.triangle,
            ),
            0,  # Inlet x-coordinate
            50,  # Outlet x-coordinate
            6.25,  # Reynolds number (used in Navier–Stokes equations)
            51.2,  # Inlet pressure
        ),
    ],
)
def test_navier_stokes_problem_pressure_inlet(
    fluid_domain,
    inlet_coordinate,
    outlet_coordinate,
    reynolds_number,
    inlet_pressure,
    plot=True,
):
    """
    Test the steady-state fluid solver with a pressure inlet boundary condition
    using the deformed fluid domain and Stokes (or Navier–Stokes) formulation.

    This test uses a flat (zero) interface displacement, applies mesh deformation,
    solves the fluid problem, and optionally visualizes the velocity and pressure fields.
    """

    # Generate interface displacement (here: flat/no displacement)
    interface_displacement = np.round(
        1
        * np.sin((np.pi * np.linspace(inlet_coordinate, outlet_coordinate, 501)) / 50),
        decimals=10,
    )

    # Deform the fluid mesh using the displacement (harmonic extension)
    deformed_mesh = mesh_deformation(
        interface_displacement, outlet_coordinate, fluid_domain, harmonic_extension=True
    )

    # Solve the fluid problem with pressure inlet
    mixed_function, interface_pressure = pressure_2d_pressure_inlet(
        deformed_mesh,
        inlet_coordinate,
        outlet_coordinate,
        reynolds_number,
        inlet_pressure,
        navier_stokes=True,  # false to use Stokes equations
    )

    def plot_interface_pressure(
        pressure: np.ndarray, x_start: float, x_end: float, title: str = ""
    ):
        """
        Plot the pressure profile along the fluid–structure interface.

        Parameters
        ----------
        pressure : np.ndarray
            Computed pressure values along the top wall (interface).
        x_start : float
            Start of the x-domain (inlet).
        x_end : float
            End of the x-domain (outlet).
        title : str
            Title for the plot.
        """
        x = np.linspace(x_start, x_end, len(pressure))
        plt.figure(figsize=(10, 4))
        plt.plot(x, pressure, label="Interface Pressure", color="blue")
        plt.xlabel("x-coordinate")
        plt.ylabel("Pressure")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    # Visualization
    if plot:
        visualize_mixed(mixed_function, deformed_mesh)
        plot_interface_pressure(
            interface_pressure,
            inlet_coordinate,
            outlet_coordinate,
            title="Interface Pressure Profile",
        )


def test_pressure_3d_pressure_inlet():
    fluid_domain_x_inlet_coordinate = 3
    fluid_domain_x_outlet_coordinate = 0
    fluid_domain_y_max_coordinate = 1
    fluid_domain_z_max_coordinate = 1
    n_x = 30
    n_y = 10
    n_z = 10

    mixed_function, interface_pressure = pressure_3d_pressure_inlet(
        fluid_domain_x_inlet_coordinate=fluid_domain_x_inlet_coordinate,
        fluid_domain_x_outlet_coordinate=fluid_domain_x_outlet_coordinate,
        fluid_domain_y_max_coordinate=fluid_domain_y_max_coordinate,
        fluid_domain_z_max_coordinate=fluid_domain_z_max_coordinate,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        reynolds_number=2,
        inlet_pressure=8,
    )

    fluid_domain = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, n_x, n_y, n_z, cell_type=dolfinx.mesh.CellType.hexahedron
    )

    # Scale the mesh geometry
    fluid_domain.geometry.x[:, 0] *= max(fluid_domain_x_inlet_coordinate, fluid_domain_x_outlet_coordinate)
    fluid_domain.geometry.x[:, 1] *= fluid_domain_y_max_coordinate
    fluid_domain.geometry.x[:, 2] *= fluid_domain_z_max_coordinate

    visualize_mixed(mixed_function, fluid_domain)
    #print(interface_pressure)

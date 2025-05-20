import dolfinx
import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import plot
import pyvista

from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import mesh_deformation
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh


@pytest.mark.parametrize(
    "interface_displacement, fluid_domain_x_max_coordinate, fluid_domain",
    [
        (
            np.round(
                1 * np.sin((np.pi * np.linspace(0, 50, 501)) / 50), decimals=10
            ),  # Synthetic sinusoidal displacement
            50,  # Max x-coordinate (domain length)
            dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD,
                np.array([[0, 0], [50, 1]]),
                [500, 10],
                cell_type=dolfinx.mesh.CellType.triangle,
            ),
        ),
    ],
)
def test_mesh_deformation(
    interface_displacement, fluid_domain_x_max_coordinate, fluid_domain
):
    """
    Test the mesh_deformation function by applying a synthetic interface displacement
    on the top wall of a fluid domain. Optionally visualize the deformed mesh.
    """

    # Applying mesh deformation
    deformed_domain = mesh_deformation(
        interface_displacement,
        fluid_domain_x_max_coordinate,
        fluid_domain,
        harmonic_extension=True,
    )

    # Verify topology and geometry
    assert isinstance(deformed_domain, dolfinx.mesh.Mesh)
    assert deformed_domain.topology.dim == fluid_domain.topology.dim

    # Extract geometry for inspection
    x = deformed_domain.geometry.x
    max_y_coord = np.max(x[:, 1])
    min_y_coord = np.min(x[:, 1])

    # Check displacement range — basic correctness
    assert max_y_coord > 1.0, "Top wall should have moved upward"
    assert min_y_coord >= 0.0, "Bottom wall should remain fixed or positive"

    print(f"Max Y coordinate after deformation: {max_y_coord:.6f}")

    # --- Optional Visualization ---
    visualize = True  # Set to False when running in CI environments

    if visualize:
        visualize_mesh(deformed_domain, title="Deformed Mesh from Interface Displacement")


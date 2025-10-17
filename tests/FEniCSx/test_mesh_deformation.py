import dolfinx
import pytest
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import mesh_deformation, mesh_deformation_3d
from VascularFlow.FEniCSx.PostProcessing.VisualizeMesh import visualize_mesh


@pytest.mark.parametrize(
    "interface_displacement, fluid_domain_x_max_coordinate, fluid_domain",
    [
        (
            np.round(
                0 * np.sin((np.pi * np.linspace(0, 50, 501)) / 50), decimals=10
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
    #assert max_y_coord > 1.0, "Top wall should have moved upward"
    #assert min_y_coord >= 0.0, "Bottom wall should remain fixed or positive"

    print(f"Max Y coordinate after deformation: {max_y_coord:.6f}")

    # --- Optional Visualization ---
    visualize = True  # Set to False when running in CI environments

    if visualize:
        visualize_mesh(deformed_domain, title="Deformed Mesh from Interface Displacement")


def test_mesh_deformation_3d():
    # Interface displacement values prescribed on the TOP face (z = Lz)
    # for a CG1 (Q1) grid with (n_y+1) × (n_x+1) nodes.
    # Ordering convention (as required by mesh_deformation_3d):
    #   y = Ly, Ly-Δy, …, 0   (rows, descending in y)
    #   within each row: x = Lx, Lx-Δx, …, 0 (descending in x)
    interface_displacement = np.array([
        0.00000000e+00, 6.42986466e-17, -1.70485754e-16, -2.33327337e-16,
        -4.24164428e-16, -2.45173459e-17, 5.43107595e-17, 0.00000000e+00,
        2.55810990e-17, 5.13539001e-02, 1.29518168e-01, 1.76299392e-01,
        1.76299392e-01, 1.29518168e-01, 5.13539001e-02, 0.00000000e+00,
        -1.83956153e-16, 1.29518168e-01, 3.28701899e-01, 4.51616223e-01,
        4.51616223e-01, 3.28701899e-01, 1.29518168e-01, 0.00000000e+00,
        1.52932103e-16, 1.76299392e-01, 4.51616223e-01, 6.23487607e-01,
        6.23487607e-01, 4.51616223e-01, 1.76299392e-01, 0.00000000e+00,
        3.95755300e-16, 1.76299392e-01, 4.51616223e-01, 6.23487607e-01,
        6.23487607e-01, 4.51616223e-01, 1.76299392e-01, 0.00000000e+00,
        3.09643928e-17, 1.29518168e-01, 3.28701899e-01, 4.51616223e-01,
        4.51616223e-01, 3.28701899e-01, 1.29518168e-01, 0.00000000e+00,
        2.39822803e-18, 5.13539001e-02, 1.29518168e-01, 1.76299392e-01,
        1.76299392e-01, 1.29518168e-01, 5.13539001e-02, 1.12495032e-17,
        0.00000000e+00, 6.25223572e-17, -1.04450521e-17, 8.46530131e-17,
        5.05841023e-19, 2.09653876e-16, 2.75157641e-17, 0.00000000e+00
    ])
    # Zero out tiny numerical noise so the boundary data is clean.
    interface_displacement[np.abs(interface_displacement) < 1e-7] = 0
    # Box size (Lx, Ly, Lz) and mesh resolution (n_x, n_y, n_z).
    # With CG1 on hexes, the top face will have (n_y+1) × (n_x+1) nodes.
    fluid_domain_x_max_coordinate=1
    fluid_domain_y_max_coordinate=1
    fluid_domain_z_max_coordinate=1
    n_x=7
    n_y=7
    n_z=7

    # Run the deformation: solves Laplace on the box with the given top-face
    # Dirichlet data and displaces the mesh in the z-direction.
    deformed_domain_3d = mesh_deformation_3d(
        interface_displacement,
        fluid_domain_x_max_coordinate,
        fluid_domain_y_max_coordinate,
        fluid_domain_z_max_coordinate,
        n_x,
        n_y,
        n_z,
    )
    # Visual inspection: render the deformed mesh.
    visualize_mesh(deformed_domain_3d, title="Deformed Mesh from Interface Displacement")

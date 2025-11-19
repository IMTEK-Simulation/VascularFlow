import dolfinx
import pytest
import numpy as np
from mpi4py import MPI

from VascularFlow.FEniCSx.MeshMovingTechnique.MeshDeformation import (
    mesh_deformation,
    mesh_deformation_3d,
    mesh_deformation_3d_rigid_elastic_rigid_channel,
)
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
    # assert max_y_coord > 1.0, "Top wall should have moved upward"
    # assert min_y_coord >= 0.0, "Bottom wall should remain fixed or positive"

    print(f"Max Y coordinate after deformation: {max_y_coord:.6f}")

    # --- Optional Visualization ---
    visualize = True  # Set to False when running in CI environments

    if visualize:
        visualize_mesh(
            deformed_domain, title="Deformed Mesh from Interface Displacement"
        )


def test_mesh_deformation_3d():
    interface_displacement = np.array([
        3.08524224e-16, 1.24402995e-14, 1.06787082e-15, -9.48998972e-16,
        6.01597103e-15, -2.06970622e-14, 2.29953555e-15, -2.34770682e-15,
        3.41781063e-16, -1.04062146e-16, -1.48760570e-15, 1.56681347e-01,
        2.66827993e-01, 2.78569369e-01, 2.59292901e-01, 2.34969635e-01,
        2.08595912e-01, 1.68381287e-01, 8.57789226e-02, 4.23468785e-15,
        0.00000000e+00, 4.70679709e-01, 8.24458953e-01, 8.76439679e-01,
        8.19521617e-01, 7.42839552e-01, 6.57434930e-01, 5.23688324e-01,
        2.62242544e-01, -4.24498727e-16, 0.00000000e+00, 7.81632304e-01,
        1.39738033e+00, 1.50417015e+00, 1.41169496e+00, 1.27987810e+00,
        1.12994683e+00, 8.92038128e-01, 4.40813798e-01, -7.51612308e-15,
        0.00000000e+00, 9.98604296e-01, 1.80736543e+00, 1.95986167e+00,
        1.84373042e+00, 1.67181601e+00, 1.47367101e+00, 1.15734146e+00,
        5.67219527e-01, -2.16164960e-15, 0.00000000e+00, 1.07559583e+00,
        1.95470957e+00, 2.12480050e+00, 2.00051901e+00, 1.81407944e+00,
        1.59822277e+00, 1.25300858e+00, 6.12396166e-01, -3.93031525e-15,
        -1.13743878e-14, 9.98604296e-01, 1.80736543e+00, 1.95986167e+00,
        1.84373042e+00, 1.67181601e+00, 1.47367101e+00, 1.15734146e+00,
        5.67219527e-01, -9.84062656e-16, -2.02824041e-15, 7.81632304e-01,
        1.39738033e+00, 1.50417015e+00, 1.41169496e+00, 1.27987810e+00,
        1.12994683e+00, 8.92038128e-01, 4.40813798e-01, 0.00000000e+00,
        1.90119800e-14, 4.70679709e-01, 8.24458953e-01, 8.76439679e-01,
        8.19521617e-01, 7.42839552e-01, 6.57434930e-01, 5.23688324e-01,
        2.62242544e-01, 4.87183194e-15, 1.91263591e-14, 1.56681347e-01,
        2.66827993e-01, 2.78569369e-01, 2.59292901e-01, 2.34969635e-01,
        2.08595912e-01, 1.68381287e-01, 8.57789226e-02, 5.86797498e-15,
        -6.08454221e-15, 9.02222925e-15, 1.78064925e-14, 1.23369740e-14,
        5.01021058e-15, 1.65113766e-16, -1.15452717e-15, 4.59077733e-16,
        -2.77109118e-15, -5.87804843e-15
    ])

    interface_displacement[np.abs(interface_displacement) < 1e-7] = 0
    channel_length = 60
    channel_width = 10
    channel_height = 1

    n_x_fluid_domain = 20
    n_y_fluid_domain = 10
    n_z_fluid_domain = 5

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
    x_max_channel_right = 18
    x_min_channel_left = 45

    deformed_domain_3d = mesh_deformation_3d_rigid_elastic_rigid_channel(
        interface_displacement,
        fluid_domain,
        channel_length,
        channel_width,
        channel_height,
        x_min_channel_left,
        x_max_channel_right,
    )
    visualize_mesh(
        deformed_domain_3d, title="Deformed Mesh from Interface Displacement"
    )

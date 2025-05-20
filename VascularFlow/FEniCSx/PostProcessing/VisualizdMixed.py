"""
Visualization utilities for mixed finite element solutions using DOLFINx and PyVista.

This script contains a function to visualize velocity and pressure fields
from a mixed finite element solution (e.g., from Navier–Stokes equations and Stokes simulation).
The velocity field is visualized using glyphs, and the pressure field is visualized as a scalar field on the mesh.
"""

import dolfinx
import numpy as np
import pyvista
from dolfinx.plot import vtk_mesh


def visualize_mixed(
    mixed_function: dolfinx.fem.Function,
    fluid_domain: dolfinx.mesh.Mesh,
):
    """
    Visualizes a mixed finite element function (e.g., velocity and pressure) using PyVista.

    Parameters:
    ----------
    mixed_function : dolfinx.fem.Function
        A function containing a mixed solution (e.g., [u, p]) where `u` is a
        vector-valued velocity field and `p` is a scalar pressure field.

    fluid_domain : dolfinx.mesh.Mesh
        The mesh corresponding to the fluid domain on which the function is defined.

    Notes:
    ------
    - The velocity field is extracted from the mixed function and visualized as glyphs (arrows).
    - The pressure field is visualized as a colored scalar field.
    - This function uses PyVista for 3D visualization and requires an active display.
    """
    uh = mixed_function.sub(0).collapse()
    ph = mixed_function.sub(1).collapse()
    # Velocity visualization
    u_topology, u_cell_types, u_geometry = vtk_mesh(uh.function_space)
    values = np.zeros((u_geometry.shape[0], 3), dtype=np.float64)
    values[:, : len(uh)] = uh.x.array.real.reshape((u_geometry.shape[0], len(uh)))
    function_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    function_grid["u"] = values
    glyphs = function_grid.glyph(orient="u", factor=0.2)
    # Background mesh (wireframe)
    fluid_domain.topology.create_connectivity(
        fluid_domain.topology.dim, fluid_domain.topology.dim
    )
    u_grid = pyvista.UnstructuredGrid(
        *vtk_mesh(fluid_domain, fluid_domain.topology.dim)
    )
    # Plot velocity
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, style="wireframe", color="k")
    u_plotter.add_mesh(glyphs)
    u_plotter.view_xy()
    u_plotter.show()
    # Plot pressure
    p_topology, p_cell_types, p_geometry = vtk_mesh(ph.function_space)
    p_grid = pyvista.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
    p_grid.point_data["pressure"] = ph.x.array.real
    p_grid.set_active_scalars("pressure")
    p_plotter = pyvista.Plotter()
    p_plotter.add_mesh(p_grid, show_edges=True)
    p_plotter.view_xy()
    p_plotter.show()

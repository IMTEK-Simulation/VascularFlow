import dolfinx
import numpy as np


def visualize_mesh(mesh: dolfinx.mesh.Mesh, title: str = "Mesh Visualization"):
    """
    Visualizes a 2D DOLFINx mesh using PyVista.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh to visualize (e.g., original or deformed fluid domain).

    title : str, optional
        Title displayed on the PyVista plot window. Default is "Mesh Visualization".
    """
    import pyvista
    from dolfinx import plot

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    # Add origin marker and label
    origin = np.array([[0.0, 0.0, 0.0]])
    plotter.add_points(origin, color='red', point_size=15.0, render_points_as_spheres=True)
    plotter.add_point_labels(origin, ["(0,0,0)"], font_size=12)
    plotter.add_mesh(grid, show_edges=True)
    plotter.show_axes()
    plotter.view_xy()
    plotter.add_title(title, font_size=12)
    plotter.show()

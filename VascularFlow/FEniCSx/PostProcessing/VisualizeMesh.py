import dolfinx


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
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.add_title(title, font_size=12)
    plotter.show()

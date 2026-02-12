import numpy as np


def flat_top_hexagonal_microfluidic_network(
    outer_radius: float,
    num_rows: int,
    num_cols: int,
):
    """
    Generate node coordinates and channel connectivity for a flat-top hexagonal microfluidic network.

    Parameters
    ----------
    outer_radius : float
        Distance between the center of a hexagon cell and any of its vertices.
        Also used as the edge length for channel spacing.
    num_rows : int
        Number of vertical hexagonal unit cells in the structure.
    num_cols : int
        Number of horizontal hexagonal unit cells in the structure.

    Returns
    -------
    nodes : np.ndarray of shape (N, 2)
        Array containing (x, y) coordinates of each network node.
    connectivity_ci : np.ndarray of shape (M, 2)
        Directed connectivity list. Each row (i, j) represents a channel
        connecting inlet node i to outlet node j.

    Notes
    -----
    * num_cols must be an odd integer.
    * The generated geometry uses a flat-top hexagonal orientation.
    * Connectivity is automatically constructed by identifying pairs of nodes
        whose separation equals the outer_radius (i.e., nearest hex neighbors).
    * Channel direction is assigned from left to right (smaller x to larger x)
        to maintain consistent inlet/outlet order.
    """

    # Compute hexagon spacing geometry
    width = 2 * outer_radius
    height = np.sqrt(3) * outer_radius

    # Horizontal lattice positions using a repeating spacing pattern
    horizontal_spacing = np.linspace(0, 0.75 * num_cols + 0.25, 3 * num_cols + 2)
    pattern = np.array([True, True, False])
    num_repeats = (len(horizontal_spacing) + 2) // 3
    mask = np.tile(pattern, num_repeats)[: len(horizontal_spacing)]
    new_horizontal_spacing = horizontal_spacing[mask]

    # Vertical lattice positions
    vertical_spacing = np.linspace(0, num_rows, 2 * num_rows + 1)

    # Assemble node coordinates
    nodes = []
    val_list = [num_rows, num_rows + 1, num_rows + 1, num_rows]
    min_val = min(val_list)

    # First left column of nodes
    y_positions_left = vertical_spacing[1::2] * height
    x_left = -outer_radius
    for y in y_positions_left:
        nodes.append((x_left, y))

    # Remaining columns, alternating number of nodes vertically
    for q in range(2 * num_cols + 2):
        val = val_list[q % 4]

        if val == min_val:
            y_positions = vertical_spacing[1::2] * height
        else:
            y_positions = vertical_spacing[::2] * height

        x = new_horizontal_spacing[q] * width
        for y in y_positions:
            nodes.append((x, y))
    # Convert to NumPy array
    nodes = np.array(nodes)
    nb_nodes = len(nodes)

    # Build edge connectivity by checking nearest neighbors
    edges = []

    for i in range(nb_nodes):
        for j in range(i + 1, nb_nodes):
            dist = np.linalg.norm(nodes[i] - nodes[j])

            # If distance equals outer_radius → they are connected
            if abs(dist - outer_radius) < 1e-6:

                # Orient edges left → right for flow direction consistency
                if nodes[i, 0] < nodes[j, 0]:
                    edges.append((i, j))
                elif nodes[j, 0] < nodes[i, 0]:
                    edges.append((j, i))
                else:
                    # If same x-direction, break tie by y-coordinate
                    if nodes[i, 1] <= nodes[j, 1]:
                        edges.append((i, j))
                    else:
                        edges.append((j, i))

        # Convert edges list to array
    connectivity_ci = np.array(edges, dtype=int)

    return nodes, connectivity_ci

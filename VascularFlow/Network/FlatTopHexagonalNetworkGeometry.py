import numpy as np


def flat_top_hexagonal_microfluidic_network(
    outer_radius: float,
    num_rows: int,
    num_cols: int,
    center: bool = False,
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
    center : bool
        True  → Hexagonal nodes + center inlet + outlets
        False → Hexagonal nodes

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
    # y_positions_left = vertical_spacing[1::2] * height
    # x_left = -outer_radius
    # for y in y_positions_left:
    #    nodes.append((x_left, y))

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

    # ----------------------------------------------------
    # Add center inlet/outlets if requested
    # ----------------------------------------------------

    if center:
        x_min = np.min(nodes[:, 0])
        x_max = np.max(nodes[:, 0])

        tol = 1e-9

        left_ids = np.where(np.abs(nodes[:, 0] - x_min) < tol)[0]
        right_ids = np.where(np.abs(nodes[:, 0] - x_max) < tol)[0]

        left_ids = left_ids[np.argsort(nodes[left_ids, 1])]
        right_ids = right_ids[np.argsort(nodes[right_ids, 1])]

        nodes_list = nodes.tolist()
        edges_list = connectivity_ci.tolist()

        # ------------------
        # LEFT CENTRAL INLET
        # ------------------

        mid = len(left_ids) // 2
        i = left_ids[mid - 1]
        j = left_ids[mid]

        p1 = nodes[i]
        p2 = nodes[j]

        merge_x = x_min - 0.45 * outer_radius
        merge_y = 0.5 * (p1[1] + p2[1])

        inlet_x = merge_x - outer_radius
        inlet_y = merge_y

        merge_node = len(nodes_list)
        inlet_node = merge_node + 1

        nodes_list.append((merge_x, merge_y))
        nodes_list.append((inlet_x, inlet_y))

        edges_list.append((inlet_node, merge_node))
        edges_list.append((merge_node, i))
        edges_list.append((merge_node, j))

        # ------------------
        # RIGHT OUTLETS
        # ------------------

        for k in range(0, len(right_ids) - 1, 2):
            i = right_ids[k]
            j = right_ids[k + 1]

            p1 = nodes[i]
            p2 = nodes[j]

            merge_x = x_max + 0.45 * outer_radius
            merge_y = 0.5 * (p1[1] + p2[1])

            outlet_x = merge_x + outer_radius
            outlet_y = merge_y

            merge_node = len(nodes_list)
            outlet_node = merge_node + 1

            nodes_list.append((merge_x, merge_y))
            nodes_list.append((outlet_x, outlet_y))

            edges_list.append((i, merge_node))
            edges_list.append((j, merge_node))
            edges_list.append((merge_node, outlet_node))

        nodes = np.array(nodes_list)
        connectivity_ci = np.array(edges_list, dtype=int)

    return nodes, connectivity_ci

# -------------------------------------------------------------------------
# Function: build_connectivity
# -------------------------------------------------------------------------
# Purpose:
# --------
# Construct the *connectivity list* for a structured 2D finite element mesh
# using four-node bilinear (Q1) elements on a rectangular grid.
#
# The connectivity tells you which **global node indices** belong to each
# element in the mesh. Each element is described by 4 nodes in a consistent
# order:
#     [bottom-left, bottom-right, top-right, top-left]
#
# Node numbering convention:
# - Numbered row-by-row from bottom to top (row-major order)
# - Each row is numbered left to right
#
# Example (n_x = 3, n_y = 3):
#   Global node numbering:
#        6 --- 7 --- 8
#        |     |     |
#        3 --- 4 --- 5
#        |     |     |
#        0 --- 1 --- 2
#
#   Connectivity for each element (0-based):
#       Element 0: [0, 1, 4, 3]
#       Element 1: [1, 2, 5, 4]
#       Element 2: [3, 4, 7, 6]
#       Element 3: [4, 5, 8, 7]
#
# Parameters:
# -----------
# n_x : int
#     Number of nodes along the x-direction (horizontal)
# n_y : int
#     Number of nodes along the y-direction (vertical)
# one_based : bool, optional (default=False)
#     If True, node indices in the connectivity list start at 1 (Fortran/MATLAB style)
#     If False, indices start at 0 (Python/C style)
#
# Returns:
# --------
# elements : list of lists
#     The connectivity list: each sub-list contains the 4 node indices for an element.
# N_nodes : int
#     Total number of global nodes in the mesh (n_x * n_y)
# -------------------------------------------------------------------------
def build_connectivity(n_x: int, n_y: int, one_based: bool = False):
    """
    Generate Q1 element connectivity for a structured 2D rectangular grid.

    Parameters
    ----------
    n_x : int
        Number of nodes in the horizontal (x) direction.
    n_y : int
        Number of nodes in the vertical (y) direction.
    one_based : bool, optional
        If True, global node numbering starts at 1.
        If False (default), numbering starts at 0.

    Returns
    -------
    elements : list[list[int]]
        List of elements; each element is a list of 4 integers representing
        the global node numbers in the order:
        [bottom-left, bottom-right, top-right, top-left].
    N_nodes : int
        Total number of nodes in the mesh (n_x * n_y).
    """
    elements = []
    ncols, nrows = n_x, n_y  # Number of nodes in each direction

    # Loop over element rows (from bottom to top)
    for ey in range(n_y - 1):
        # Loop over element columns (from left to right)
        for ex in range(n_x - 1):
            # Compute the global node indices for the current element's corners
            bl = ey * ncols + ex  # bottom-left
            br = bl + 1  # bottom-right
            tl = bl + ncols  # top-left
            tr = tl + 1  # top-right

            conn0 = [bl, br, tr, tl]  # local order (0-based)

            # Adjust to 1-based indexing if requested
            if one_based:
                conn = [c + 1 for c in conn0]
            else:
                conn = conn0

            elements.append(conn)

    N_nodes = n_x * n_y
    return elements, N_nodes


# -------------------------------------------------------------------------
# Function: build_connectivity_acm
# -------------------------------------------------------------------------
# Purpose:
# --------
# Construct the *degree-of-freedom (DOF) connectivity list* for a structured
# 2D finite element mesh using acm elements on a rectangular grid.
#
# This function extends the standard node-based connectivity by accounting for
# multiple degrees of freedom per node e.g., 3 DOFs for vector-valued problems.
#
# Example (n_x = 3, n_y = 3, dofs_per_node = 3):
# ----------------------------------------------
#   Global node numbering:
#        6 --- 7 --- 8
#        |     |     |
#        3 --- 4 --- 5
#        |     |     |
#        0 --- 1 --- 2
#
#   Node connectivity for each element (0-based):
#       Element 0: [0, 1, 4, 3]
#       Element 1: [1, 2, 5, 4]
#       Element 2: [3, 4, 7, 6]
#       Element 3: [4, 5, 8, 7]
#
#   DOF connectivity for Element 0 (3 DOFs/node):
#       [0, 1, 2, 3, 4, 5, 12, 13, 14, 9, 10, 11]
# -------------------------------------------------------------------------
def build_connectivity_acm(
    n_x: int, n_y: int, dofs_per_node: int = 3, one_based: bool = False
):
    """
    Parameters
    ----------
    n_x : int
        Number of nodes in the horizontal (x) direction.
    n_y : int
        Number of nodes in the vertical (y) direction.
    dofs_per_node : int, optional
        Number of degrees of freedom per node (default: 3).
    one_based : bool, optional
        If True, use 1-based DOF numbering (for Fortran/MATLAB).
        If False (default), use 0-based numbering (for Python).

    Returns
    -------
    elements_dof : list[list[int]]
        Connectivity list of global DOF indices for each element.
        Each sublist has length = 4 * dofs_per_node.
    N_nodes : int
        Total number of global nodes (n_x * n_y).
    N_dofs : int
        Total number of global DOFs (N_nodes * dofs_per_node).
    """

    # --- Step 1: build standard 4-node Q1 connectivity ---
    elements, N_nodes = build_connectivity(n_x, n_y, one_based=False)
    # --- Step 2: expand to DOF connectivity ---
    elements_dof = []

    for conn in elements:
        dofs = []
        for node in conn:
            # Each node contributes 'dofs_per_node' global DOFs
            base = node * dofs_per_node
            # DOF indices for this node
            dofs.extend([base + d for d in range(dofs_per_node)])
        elements_dof.append(dofs)
    # --- Step 3: total number of DOFs ---
    N_dofs = N_nodes * dofs_per_node
    # --- Step 4: convert to 1-based indexing if requested ---
    if one_based:
        elements_dof = [[d + 1 for d in elem] for elem in elements_dof]

    return elements_dof, N_nodes, N_dofs
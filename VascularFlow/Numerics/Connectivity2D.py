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

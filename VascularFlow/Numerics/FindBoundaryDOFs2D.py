# -----------------------------------------------------------------------------
# boundary_dofs_2d
# -----------------------------------------------------------------------------
# In a finite element method (FEM) mesh, we often need to identify the degrees
# of freedom (DOFs) that lie along the boundaries of the domain, so that we can
# apply Dirichlet or Neumann boundary conditions.
#
# This helper function automates that process for a structured 2D grid of
# bilinear Q1 elements (4-node quadrilaterals). The mesh nodes are numbered
# in a row-major ordering:
#
#   y=height
#   ↑
#   6 --- 7 --- 8
#   |     |     |
#   3 --- 4 --- 5
#   |     |     |
#   0 --- 1 --- 2   → x=length
#
# Example:
#   - "bottom"  → DOFs [0, 1, 2]
#   - "top"     → DOFs [6, 7, 8]
#   - "left"    → DOFs [0, 3, 6]
#   - "right"   → DOFs [2, 5, 8]
#
# The function supports a single position string or a list of strings. It
# returns a tuple of DOF lists so it integrates naturally with boundary
# condition functions (e.g. applying Dirichlet constraints on multiple sides).
#
# Usage:
#   boundary_dofs_2d(3, 3, "bottom")
#   → ([0, 1, 2],)
#
#   boundary_dofs_2d(3, 3, ["bottom", "right"])
#   → ([0, 1, 2], [2, 5, 8])
# -----------------------------------------------------------------------------


from typing import Union, Sequence


def boundary_dofs_2d(
    n_x: int, n_y: int, positions: Union[str, Sequence[str]]
) -> tuple[list[int], ...]:
    """
    Return DOF indices for one or more boundaries on a structured n_x × n_y node grid.

    Node numbering (0-based): row-major from bottom-left, left→right, bottom→top.

    Parameters
    ----------
    n_x : int
        Number of nodes in x-direction.
    n_y : int
        Number of nodes in y-direction.
    positions : str or sequence of str
        Boundary positions (case-insensitive, spaces ignored).
        Each entry can be one of:
            - "y=0", "bottom"
            - "y=height", "top"
            - "x=0", "left"
            - "x=length", "right"

    Returns
    -------
    tuple[list[int], ...]
        A tuple of lists of DOF indices. If multiple positions are given,
        one list is returned per position, in order.
    """

    def single_boundary(pos: str) -> list[int]:
        key = pos.lower().replace(" ", "")
        if key in ("y=0", "bottom"):
            return list(range(n_x))
        if key in ("y=height", "top"):
            start = (n_y - 1) * n_x
            return list(range(start, start + n_x))
        if key in ("x=0", "left"):
            return [r * n_x for r in range(n_y)]
        if key in ("x=length", "right"):
            return [r * n_x + (n_x - 1) for r in range(n_y)]
        raise ValueError(f"Unknown boundary position: {pos!r}")

    # Allow single string or list/tuple of strings
    if isinstance(positions, str):
        return (single_boundary(positions),)
    else:
        return tuple(single_boundary(p) for p in positions)

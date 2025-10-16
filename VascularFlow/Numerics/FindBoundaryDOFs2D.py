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


from typing import Union, Sequence, Tuple, List, Optional


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


# -----------------------------------------------------------------------------
# boundary_dofs_acm_2d
# -----------------------------------------------------------------------------
#   y = height
#   ↑
#   6 --- 7 --- 8
#   |     |     |
#   3 --- 4 --- 5
#   |     |     |
#   0 --- 1 --- 2   → x = length
#
# Supported position keywords:
#   - "y=0" or "bottom"   → bottom edge
#   - "y=height" or "top" → top edge
#   - "x=0" or "left"     → left edge
#   - "x=length" or "right" → right edge
# Example (3×3 nodes, 3 DOFs per node):
#   - "bottom" with all components:
#         → [0,1,2, 3,4,5, 6,7,8]
#   - "bottom" with only component 0 (u):
#         → [0, 3, 6]
#   - ["bottom", "right"] boundaries:
#         → ([0,1,2,3,4,5,6,7,8], [2,3,4,5,6,7,8])
#
# -----------------------------------------------------------------------------
def boundary_dofs_acm_2d(
    n_x: int,
    n_y: int,
    positions: Union[str, Sequence[str]],
    *,
    dofs_per_node: int = 3,
    components: Optional[Sequence[int]] = None,
    one_based: bool = False,
) -> Tuple[List[int], ...]:
    """
    Return global DOF indices for one or more boundaries on a structured n_x × n_y node grid,
    expanding node indices to DOF indices for multi-DOF nodes.

    Node numbering (0-based): row-major from bottom-left, left→right, bottom→top.

    Parameters
    ----------
    n_x : int
        Number of nodes in x-direction.
    n_y : int
        Number of nodes in y-direction.
    positions : str or sequence[str]
        Boundary positions (case-insensitive, spaces ignored). Each can be one of:
            - "y=0", "bottom"
            - "y=height", "top"
            - "x=0", "left"
            - "x=length", "right"
    dofs_per_node : int, default=3
        Number of DOFs per node (e.g., 3 for vector fields).
    components : sequence[int] or None, default=None
        Which within-node components to include. If None, includes all `range(dofs_per_node)`.
        Example: components=[0] returns only the first DOF per node; [0,2] returns the 1st and 3rd.
    one_based : bool, default=False
        If True, return 1-based DOF indices; else 0-based.

    Returns
    -------
    tuple[list[int], ...]
        A tuple with one list per position, in the same order as `positions`.

    Examples
    --------
    # 3x3 nodes, 3 dof per node, bottom boundary, all components:
    #boundary_dofs_2d(3, 3, "bottom", dofs_per_node=3)
    ([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # same but only component 1 (zero-based) per node:
    #boundary_dofs_2d(3, 3, "bottom", dofs_per_node=3, components=[1])
    ([1, 4, 7])

    # multiple boundaries:
    #boundary_dofs_2d(3, 3, ["bottom", "right"], dofs_per_node=3)
    ([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8])
    """

    def single_boundary_nodes(pos: str) -> List[int]:
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

    # which components within each node to include
    if components is None:
        comps = list(range(dofs_per_node))
    else:
        comps = list(components)
        if any(c < 0 or c >= dofs_per_node for c in comps):
            raise ValueError(f"'components' must be in [0, {dofs_per_node-1}]")

    def expand_nodes_to_dofs(node_ids: Sequence[int]) -> List[int]:
        dofs: List[int] = []
        for node in node_ids:
            base = node * dofs_per_node
            for c in comps:
                dof = base + c
                dofs.append(dof + 1 if one_based else dof)
        return dofs

    # Allow single string or list/tuple of strings
    if isinstance(positions, str):
        nodes = single_boundary_nodes(positions)
        return (expand_nodes_to_dofs(nodes),)
    else:
        result: List[List[int]] = []
        for p in positions:
            nodes = single_boundary_nodes(p)
            result.append(expand_nodes_to_dofs(nodes))
        return tuple(result)

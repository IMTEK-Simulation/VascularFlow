"""
First derivative of an arbitrary array using the finite difference method.
Used in the conservation of momentum in Navier–Stokes equations (∂p/∂x = -ReSt/H ∂Q/∂t - 6/5 Re/H ∂/∂x(Q2/H) - 12Q/H3)
to take the first derivative of Q2/H.

Notes:
- This is a second-order accurate scheme for all points.
"""

import numpy as np


def array_first_derivative(array: np.ndarray, mesh_nodes: np.ndarray) -> np.ndarray:
    """
    This function estimates the derivative using:
    - Central difference for interior points
    - Forward difference at the first point
    - Backward difference at the last point

    Parameters
    ----------
    array : np.ndarray
        1D array of function values evaluated at uniformly spaced points.
    mesh_nodes : np.ndarray
        Global mesh node positions.

    Returns
    -------
    derivative : np.ndarray
        1D array of the same shape as `array`, containing the first derivative
        approximations at each point.
    """
    derivative = np.zeros_like(array)
    dx_e = (mesh_nodes[-1] - mesh_nodes[0]) / (len(mesh_nodes) - 1)

    # Central difference for interior points
    derivative[1:-1] = (array[2:] - array[:-2]) / (2 * dx_e)

    # Forward difference for the first point
    derivative[0] = (-3 * array[0] + 4 * array[1] - array[2]) / (2 * dx_e)

    # Backward difference for the last point
    derivative[-1] = (3 * array[-1] - 4 * array[-2] + array[-3]) / (2 * dx_e)

    return derivative

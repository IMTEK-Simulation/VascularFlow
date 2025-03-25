import numpy as np


def initialize_flow_arrays(nb_nodes: int):
    """
    Initializes flow-related arrays based on the spatial grid x_n.

    Parameters:
        nb_nodes (int): Number of spatial grid points.

    Returns:
        dict: A dictionary containing initialized arrays.
    """
    size = nb_nodes
    return {
        "h_n_1": np.ones(size),
        "h_n": np.ones(size),
        "h_star": np.ones(size),
        "h_new": np.ones(size),
        "q_n_1": np.ones(size),
        "q_n": np.ones(size),
        "q_star": np.ones(size),
        "q_new": np.ones(size),
        "p": np.zeros(size),
        "p_inner": np.zeros(size),
    }

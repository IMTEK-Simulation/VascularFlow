def initialize_constants():
    """
    Initializes scalar constants related to the simulation.

    Returns:
        dict: A dictionary containing scalar simulation parameters.
    """
    return {
        "epsilon": 0.02,
        "Re": 7.5,
        "St": 0.68,
        "Beta": 35156.24,
        "relax": 0.00003,
        "q0": 1,
    }

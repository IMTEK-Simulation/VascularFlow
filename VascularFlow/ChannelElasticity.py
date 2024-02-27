def independent_rings(pressure_n, initial_area, ring_modulus):
    """
    Calculates the area of a tube under the assumption of independent rings.

    Parameters
    ----------
    pressure_n : float
        The pressure along the tube.
    initial_area : float
        The initial area of the tube.
    ring_modulus : float
        The ring modulus of the tube.

    Returns
    -------
    area_n : float
        The area of the tube.
    """
    return initial_area * (1 + (pressure_n / ring_modulus))

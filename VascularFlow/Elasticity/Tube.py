import numpy as np


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


def linear_pressure_area(area_e_new, initial_area, inlet_pressure, ring_modulus):
    """
        Calculates the pressure of each node along the tube under the assumption of independent rings.

        Parameters
        ----------
        area_e_new : ndarray
            The new area of each element.
        initial_area : ndarray
            The initial area of the tube.
        inlet_pressure : float
            The inlet pressure of the tube
        ring_modulus : float
            The ring modulus of the tube.

        Returns
        -------
        pressure : ndarray
            the pressure of each node along the tube.
        """
    return np.append(inlet_pressure, inlet_pressure + ring_modulus * ((area_e_new / initial_area) - 1))

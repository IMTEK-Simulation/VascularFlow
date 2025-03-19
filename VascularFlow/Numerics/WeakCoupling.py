import numpy as np

from VascularFlow.Elasticity.Tube import independent_rings
from VascularFlow.Flow.ChannelFlow import pressure_change


def solve_weakly_coupled_channel(positions_n, initial_area, ring_modulus, inlet_pressure, flow_rate, density,
                                 kinematic_viscosity,
                                 alpha=4 / 3, l1norm_target=1e-20):
    """
    Solves the weakly coupled system of equations for a channel flow.

    Parameters
    ----------
    positions_n : np.ndarray
        The nodal positions along the channel.
    initial_area : np.ndarray
        The initial cross-sectional area of the tube.
    ring_modulus : float
        The ring modulus of the tube.
    inlet_pressure : float
        The pressure at the inlet.
    flow_rate : float
        The flow rate at the input of the channel.
    density : float
        The density of the fluid.
    kinematic_viscosity : float
        The kinematic viscosity of the fluid.
    alpha : float
        Momentum correction factor. (Default: 4/3)

    Returns
    -------
    area_e : np.ndarray
        The cross-sectional areas.
    pressure_n : np.ndarray
        The pressure across the pipe for each nodal position along the pipe.
    """
    area_e = np.ones(len(positions_n) - 1) * initial_area
    l1norm = 1 + l1norm_target
    while l1norm > l1norm_target:
        old_area_e = area_e.copy()

        pressure_n = inlet_pressure + pressure_change(positions_n, area_e, flow_rate, density, kinematic_viscosity,
                                                      alpha)
        area_e = independent_rings((pressure_n[1:] + pressure_n[:-1])/2,
                                   initial_area, ring_modulus)

        l1norm = np.linalg.norm(area_e - old_area_e)

    pressure_n = inlet_pressure + pressure_change(positions_n, area_e, flow_rate, density, kinematic_viscosity, alpha)

    return area_e, pressure_n

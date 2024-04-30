import numpy as np
from scipy.integrate import solve_ivp

from VascularFlow.ChannelFlow_Unsteady import dAdt
from VascularFlow.ChannelFlow_Unsteady import dQdt


def propagate(time_interval, positions_n, initial_area_e, initial_flow_rate_n, A0, density, kinematic_viscosity,
              ring_modulus, alpha=4/3, method='RK45'):
    """
    Calculates the time rate of flow rate changes for each node across the channel.

    Parameters
    ----------
    time_interval : float
        Integration interval.
    positions_n : np.ndarray
        The nodal positions along the channel.
    initial_area_e : np.ndarray
        the cross-sectional area for each element across the channel.
    initial_flow_rate_n : np.ndarray
        The flow rate at each node along the channel.
    A0 : float
        the area of the unstressed channel.
    density : float
        The fluid density
    kinematic_viscosity : float
        The fluid kinematic viscosity
    ring_modulus : float
        The ring modulus of the tube.
    alpha : float
        Momentum correction factor. (Default: 4/3)
    method : str, optional
        Integration method (see documentation of `solve_ivp`).
        Default: 'RK45'

    Returns
    -------
        the time rate of flow rate changes for each node across the channel.
    """

    nb_elements = positions_n.size-1

    def dydt(t, y):
        current_area_e = y[:nb_elements]
        current_flow_rate_n = y[nb_elements:]

        darea_dt_e = dAdt(positions_n, current_flow_rate_n)
        dflow_rate_dt_n = dQdt(positions_n, current_area_e, current_flow_rate_n, A0, density, kinematic_viscosity,
                               ring_modulus, alpha)

        return np.append(darea_dt_e, dflow_rate_dt_n)

    initial_y = np.append(initial_area_e, initial_flow_rate_n)
    final_y = solve_ivp(dydt, [0, time_interval], initial_y, t_eval=[time_interval], method='RK45')
    print(final_y.message)
    # print(final_y.t)
    final_area_e = final_y.y[:nb_elements]
    # print(final_area_e)
    final_flow_rate_n = final_y.y[nb_elements:]
    # print(final_flow_rate_n)
    return final_area_e, final_flow_rate_n

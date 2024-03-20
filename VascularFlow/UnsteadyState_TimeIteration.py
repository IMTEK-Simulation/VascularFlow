import numpy as np

from VascularFlow.ChannelFlow_Unsteady import dAdt
from VascularFlow.ChannelFlow_Unsteady import new_area_e
from VascularFlow.ChannelFlow_Unsteady import dQdt
from VascularFlow.ChannelFlow_Unsteady import new_flow_rate_n


def forward_euler_method(positions_n, area_e, A0, flow_rate_n, dt, density,
                         kinematic_viscosity, ring_modulus):
    """
        Solves the system of equations for a time dependent channel flow.

        Parameters
        ----------
        positions_n : np.ndarray
            The nodal positions along the channel.
        area_e : np.ndarray
            The area at each element in the previous time step.
        A0 : float
            the area of the unstressed channel.
        flow_rate_n : np.ndarray
            The flow rate at each element in the previous time step.
        dt : float
            The time step
        density : float
            The density of the fluid.
        kinematic_viscosity : float
            The kinematic viscosity of the fluid.
        ring_modulus : float
            The ring modulus of the tube.

        Returns
        -------
        area_e : np.ndarray
            The cross-sectional area at each element in the final time step.
        flow_rate_n : np.ndarray
            The flow rate across the pipe for each nodal position along the pipe.
        """
    t = 0  # The initial value for time to iterate
    tend = 500 * dt  # The final time
    while t < tend:
        t = t + dt
        area_e_new = new_area_e(area_e, dt, positions_n, flow_rate_n)
        flow_rate_n_new = new_flow_rate_n(positions_n, area_e, flow_rate_n, A0, dt, density,
                                          kinematic_viscosity, ring_modulus)
        area_e = area_e_new
        flow_rate_n = flow_rate_n_new
    return area_e, flow_rate_n

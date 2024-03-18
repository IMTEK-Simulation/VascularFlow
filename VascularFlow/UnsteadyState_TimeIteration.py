import numpy as np

from VascularFlow.ChannelFlow_Unsteady import area_change
from VascularFlow.ChannelFlow_Unsteady import flow_rate_change
from VascularFlow.ChannelElasticity import linear_pressure_area


def unsteady_state_channel(positions_n, area_e, initial_area, flow_rate, inlet_pressure, pressure, dt, density,
                           kinematic_viscosity, ring_modulus,
                           alpha=4 / 3):
    """
        Solves the system of equations for a time dependent channel flow.

        Parameters
        ----------
        positions_n : np.ndarray
            The nodal positions along the channel.
        area_e : np.ndarray
            The area at each element in the past time step.
        initial_area : float
            The initial area at each element
        flow_rate : np.ndarray
            The flow rate at each element in the past time.
        inlet_pressure : float
            The inlet pressure of the tube
        pressure : np.ndarray
            The pressure at each node in the past time.
        dt : float
            The time step
        density : float
            The density of the fluid.
        kinematic_viscosity : float
            The kinematic viscosity of the fluid.
        ring_modulus : float
            The ring modulus of the tube.
        alpha : float
            Momentum correction factor. (Default: 4/3)

        Returns
        -------
        area_e : np.ndarray
            The cross-sectional areas.
        flow_rate_n : np.ndarray
            The flow rate across the pipe for each nodal position along the pipe.
        pressure_n : np.ndarray
            The pressure across the pipe for each nodal position along the pipe.
        """
    t = 0  # The initial value for time to iterate
    tend = 5000 * dt  # The final time
    while t < tend:
        t = t + dt
        area_e_new = area_change(positions_n, area_e, dt, flow_rate)
        area_e_new[0] = initial_area  # Dirichlet boundary condition
        flow_rate_new = flow_rate_change(positions_n, area_e, dt, flow_rate, pressure,
                                         density, kinematic_viscosity, alpha)
        pressure_new = linear_pressure_area(area_e_new, initial_area, inlet_pressure, ring_modulus)
        pressure_new[0] = inlet_pressure  # Dirichlet boundary condition
        area_e = area_e_new
        flow_rate = flow_rate_new
        pressure = pressure_new
    return area_e, flow_rate, pressure

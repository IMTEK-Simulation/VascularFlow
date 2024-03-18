import numpy as np
import pytest

from VascularFlow.UnsteadyState_TimeIteration import unsteady_state_channel


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4, 5],
                                              [0, 0.7, 2.3, 3, 4.5, 5]])
def test_unsteady_state_time_iteration(node_positions_n, plot=True):
    """Test the solution of time dependent flow through an elastic tube."""
    node_positions_n = np.asanyarray(node_positions_n)
    nb_elements = len(node_positions_n)-1
    nb_nodes = len(node_positions_n)
    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    initial_area = np.pi * (tube_radius ** 2)
    area_e = np.full(nb_elements, initial_area)
    inlet_flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries
    flow_rate = np.full(nb_nodes, inlet_flow_rate)
    inlet_pressure = 2000  # inlet blood pressure Pa
    pressure = np.full(nb_nodes, inlet_pressure)
    dt = 0.00000001
    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s
    ring_modulus = 21.2e3  # tube wall elasticity coefficient (in Pa)

    area_e, flow_rate, pressure = unsteady_state_channel(node_positions_n, area_e, initial_area, flow_rate,
                                                         inlet_pressure, pressure,
                                                         dt, density, kinematic_viscosity, ring_modulus, alpha=4 / 3)
    print(area_e, flow_rate, pressure)
    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot((node_positions_n[1:] + node_positions_n[:-1])/2, area_e, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Area (m2)')
        #plt.subplot(212)
        #plt.plot(node_positions_n, flow_rate, 'kx-')
        #plt.xlabel('Position along tube (m)')
        #plt.ylabel('Flow rate (m^3/s)')
        #plt.subplot(213)
        #plt.plot(node_positions_n, pressure, 'kx-')
        #plt.xlabel('Position along tube (m)')
        #plt.ylabel('pressure (Pa)')
        #plt.tight_layout()
        plt.show()

import numpy as np
import pytest

from VascularFlow.UnsteadyState_TimeIteration import forward_euler_method


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4],
                                              [0, 0.5, 1.5, 3.5, 4]])
def test_unsteady_state_time_iteration(node_positions_n, plot=True):
    """Test the solution of time dependent flow through an elastic tube."""
    node_positions_n = np.asanyarray(node_positions_n)
    nb_elements = len(node_positions_n)-1
    nb_nodes = len(node_positions_n)
    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)
    inlet_flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries
    flow_rate_init = np.full(nb_nodes, inlet_flow_rate)
    dt = 0.00000001
    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s
    ring_modulus = 21.2e3  # tube wall elasticity coefficient (in Pa)

    area_e, flow_rate_n = forward_euler_method(node_positions_n, area_e_init, tube_area, flow_rate_init,
                                               dt, density, kinematic_viscosity, ring_modulus)
    print(area_e, flow_rate_n)
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

import numpy as np
import pytest

from VascularFlow.UnsteadyState_TimeIteration import propagate


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4]])
def test_unsteady_state_time_iteration(node_positions_n, plot=True):
    """Test the solution of time dependent flow through an elastic tube."""
    node_positions_n = np.asanyarray(node_positions_n)
    nb_nodes = node_positions_n.size
    nb_elements = nb_nodes-1
    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    A0 = np.pi * (tube_radius ** 2)
    initial_area_e = np.full(nb_elements, A0)
    inlet_flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries
    initial_flow_rate_n = np.zeros(nb_nodes - 1)
    initial_flow_rate_n = np.append(inlet_flow_rate, initial_flow_rate_n)
    time_interval = 5
    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s
    ring_modulus = 21.2e3  # tube wall elasticity coefficient (in Pa)

    final_area, final_flow_rate = propagate(time_interval, node_positions_n, initial_area_e, initial_flow_rate_n, A0,
                                            density, kinematic_viscosity, ring_modulus, alpha=4/3, method='RK45')
    print(final_area, final_flow_rate)
    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot((node_positions_n[1:] + node_positions_n[:-1])/2, final_area, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Area (m2)')
        plt.show()

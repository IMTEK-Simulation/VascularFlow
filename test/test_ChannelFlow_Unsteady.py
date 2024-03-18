import numpy as np
import pytest

from VascularFlow.ChannelFlow_Unsteady import area_change
from VascularFlow.ChannelFlow_Unsteady import flow_rate_change


@pytest.mark.parametrize('node_positions_n', [[0, 0.25, 0.5, 0.75, 1],
                                              [0, 0.125, 0.5, 0.9, 1]])
def test_area_change(node_positions_n, plot=False):
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1
    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)  # cross-sectional area of the pipe for each element along the pipe
    flow_rate_init = np.full(nb_nodes, 4e-7)
    dt = 0.0001
    new_area = area_change(node_positions_n, area_e_init, dt, flow_rate_init)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(node_positions_n[1:], new_area, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('area (m2)')
        plt.show()


@pytest.mark.parametrize('node_positions_n', [[0, 0.25, 0.5, 0.75, 1],
                                              [0, 0.125, 0.5, 0.9, 1]])
def test_flow_rate_change(node_positions_n, plot=True):
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1
    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)  # cross-sectional area of the pipe for each element along the pipe
    flow_rate_init = np.full(nb_nodes, 4e-7)
    pressure_init = np.full(nb_nodes, 1)
    dt = 0.0001
    density = 1
    kinematic_viscosity = 1
    new_flow_rate = flow_rate_change(node_positions_n, area_e_init, dt, flow_rate_init, pressure_init, density, kinematic_viscosity)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(node_positions_n[:], new_flow_rate, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('flow rate (m3/s)')
        plt.show()

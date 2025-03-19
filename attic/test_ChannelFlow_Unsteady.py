import numpy as np
import pytest

from VascularFlow.Flow.ChannelFlow_Unsteady import dAdt
from VascularFlow.Flow.ChannelFlow_Unsteady import new_area_e
from VascularFlow.Flow.ChannelFlow_Unsteady import dQdt
from VascularFlow.Flow.ChannelFlow_Unsteady import new_flow_rate_n


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4],
                                              [0, 0.5, 1.5, 3.5, 4]])
def test_dAdt(node_positions_n):
    nb_nodes = len(node_positions_n)
    flow_rate_init = np.full(nb_nodes, 4e-7)
    time_changes_area_e = dAdt(node_positions_n, flow_rate_init)
    print(time_changes_area_e)


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4],
                                              [0, 0.5, 1.5, 3.5, 4]])
def test_new_area_e(node_positions_n, plot=False):
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1
    tube_radius = 1.8e-3
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)
    dt = 0.001
    flow_rate_init = np.full(nb_nodes, 4e-7)
    area_e_new = new_area_e(area_e_init, dt, node_positions_n, flow_rate_init)
    print(area_e_new)


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4],
                                              [0, 0.5, 1.5, 3.5, 4]])
def test_dQdt(node_positions_n, plot=False):
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1
    tube_radius = 1.8e-3
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)
    flow_rate_init = np.full(nb_nodes, 4e-7)
    density = 1050
    kinematic_viscosity = 3.2e-6
    ring_modulus = 21.2e3
    time_changes_flow_rate_n = dQdt(node_positions_n, area_e_init, flow_rate_init, tube_area, density,
                                    kinematic_viscosity, ring_modulus, alpha=4 / 3)
    print(time_changes_flow_rate_n)


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4],
                                              [0, 0.5, 1.5, 3.5, 4]])
def test_new_flow_rate_n(node_positions_n, plot=False):
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1
    tube_radius = 1.8e-3
    tube_area = np.pi * (tube_radius ** 2)
    area_e_init = tube_area * np.ones(nb_elements)
    dt = 0.001
    flow_rate_init = np.full(nb_nodes, 4e-7)
    density = 1050
    kinematic_viscosity = 3.2e-6
    ring_modulus = 21.2e3
    flow_rate_n_new = new_flow_rate_n(node_positions_n, area_e_init, flow_rate_init, tube_area, dt, density,
                                      kinematic_viscosity, ring_modulus)
    print(flow_rate_n_new)







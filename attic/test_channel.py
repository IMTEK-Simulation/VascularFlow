import numpy as np
import pytest

from VascularFlow.Flow.ChannelFlow import pressure_change


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4, 5],
                                              [0, 0.7, 2.3, 3, 4.5, 5]])
def test_homogeneous_tube(node_positions_n, plot=False):
    """Test pressure change across a homogeneous tube. It should be a linear function."""
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1

    flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries

    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    tube_area = np.pi * (tube_radius ** 2)
    area_e = tube_area * np.ones(nb_elements)  # cross-sectional area of the pipe for each element along the pipe

    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s

    pressure_n = pressure_change(node_positions_n, area_e, flow_rate, density, kinematic_viscosity)

    analytic_pressure_n = - 8 * np.pi * kinematic_viscosity * flow_rate * density / tube_area ** 2 * np.array(node_positions_n)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(node_positions_n, analytic_pressure_n, 'r-', lw=4)
        plt.plot(node_positions_n, pressure_n, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Pressure (Pa)')
        plt.show()

    np.testing.assert_allclose(pressure_n, analytic_pressure_n)


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4, 5],
                                              [0, 0.7, 2.3, 3, 4.5, 5]])
def test_linear_tube(node_positions_n, plot=True):
    """Test pressure change across a homogeneous tube. It should be a linear function."""
    node_positions_n = np.array(node_positions_n)
    nb_nodes = len(node_positions_n)
    nb_elements = nb_nodes - 1

    flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries

    tube_radius_inlet = 1.8e-3  # unstressed tube radius (in m)
    tube_radius_outlet = 1.0e-3  # unstressed tube radius (in m)
    tube_area_inlet = np.pi * (tube_radius_inlet ** 2)
    tube_area_outlet = np.pi * (tube_radius_outlet ** 2)
    area_n = tube_area_inlet + (tube_area_outlet - tube_area_inlet) * (node_positions_n - node_positions_n[0]) / (node_positions_n[-1] - node_positions_n[0])
    area_e = (area_n[1:] + area_n[:-1]) / 2

    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s

    pressure_n = pressure_change(node_positions_n, area_e, flow_rate, density, kinematic_viscosity)

    # TODO: Analytic solution for linearly changing area
    #analytic_pressure_n = - 8 * np.pi * kinematic_viscosity * flow_rate * density / tube_area ** 2 * np.array(node_positions_n)

    if plot:
        import matplotlib.pyplot as plt
        #plt.plot(node_positions_n, analytic_pressure_n, 'r-', lw=4)
        plt.subplot(211)
        plt.plot(node_positions_n, area_n, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Area (m^2)')
        plt.subplot(212)
        plt.plot(node_positions_n, pressure_n, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Pressure (Pa)')
        plt.tight_layout()
        plt.show()

    #np.testing.assert_allclose(pressure_n, analytic_pressure_n)

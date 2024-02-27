import numpy as np
import pytest

from VascularFlow.WeakCoupling import solve_weakly_coupled_channel


@pytest.mark.parametrize('node_positions_n', [[0, 1, 2, 3, 4, 5],
                                              [0, 0.7, 2.3, 3, 4.5, 5]])
def test_weak_coupling(node_positions_n, plot=True):
    """Test weakly-coupled flow/elastic solution."""
    node_positions_n = np.array(node_positions_n)

    flow_rate = 4e-7  # blood flow rate is 3.0~26 ml/min in arteries

    ring_modulus = 1000  # tube wall elasticity coefficient (in Pa)

    tube_radius = 1.8e-3  # unstressed tube radius (in m)
    tube_area = np.pi * (tube_radius ** 2)

    density = 1050  # average blood density kg/m^3
    kinematic_viscosity = 3.2e-6  # blood kinematic viscosity m^2/s

    inlet_pressure = 2000  # inlet blood pressure Pa

    area_e, pressure_n = solve_weakly_coupled_channel(
        node_positions_n,
        tube_area,
        ring_modulus,
        inlet_pressure,
        flow_rate,
        density,
        kinematic_viscosity)

    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(node_positions_n, pressure_n, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Pressure (Pa)')
        plt.subplot(212)
        plt.plot((node_positions_n[1:] + node_positions_n[:-1])/2, area_e, 'kx-')
        plt.xlabel('Position along tube (m)')
        plt.ylabel('Area (m^2)')
        plt.tight_layout()
        plt.show()

    #np.testing.assert_allclose(pressure_n, analytic_pressure_n)

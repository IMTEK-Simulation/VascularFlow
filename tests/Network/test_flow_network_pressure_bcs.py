import numpy as np


from VascularFlow.Network.FlowNetworkPressureBCs import (
    flow_network_1d_pressure_boundary_condition,
)
from VascularFlow.Network.FlatTopHexagonalNetworkGeometry import (
    flat_top_hexagonal_microfluidic_network,
)
from VascularFlow.Network.PostProcessing import (
    plot_flow_network,
)


def test_flow_network_pressure_bcs():
    nodes, connectivity_ci = flat_top_hexagonal_microfluidic_network(1, 3, 5)

    boundary_nodes = np.array([0, 1, 2, 42, 43, 44])
    boundary_pressure_b = np.array([10, 15, 20, 0, 1, 2])
    resistance = 1.0

    # -------------------------------------------------------------------------
    # Act: solve for the node pressures
    # -------------------------------------------------------------------------
    pressure = flow_network_1d_pressure_boundary_condition(
        connectivity_ci=connectivity_ci,
        boundary_nodes=boundary_nodes,
        boundary_pressures=boundary_pressure_b,
        resistance=resistance,
    )

    print("\nComputed Pressure Values:")
    print(pressure)

    inlet_c, outlet_c = np.transpose(connectivity_ci)
    plot_flow_network(
        pressure_n=pressure,
        inlet_c=inlet_c,
        outlet_c=outlet_c,
        nodes=nodes,  # only if you have geometry coordinates
        title="Flow + Pressure Distribution",
    )

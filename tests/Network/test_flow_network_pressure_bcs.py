import numpy as np
import pytest

from VascularFlow.Network.FlowNetworkPressureBCs import (
    flow_network_1d_pressure_boundary_condition,
)
from VascularFlow.Network.FlatTopHexagonalNetworkGeometry import (
    flat_top_hexagonal_microfluidic_network,
)
from VascularFlow.Network.ChannelLaws import channel_law_rigid, channel_law_center
from VascularFlow.Network.PostProcessing import (
    plot_flow_network,
    plot_pressure_nodes_network,
    plot_p_bar_network,
    plot_conductance_network,
    plot_velocity_pressure_network,
    plot_esr_outlet_flow_rate_ratio,
)

dynamic_viscosity = 1e-3

# Network channels dimensions
H0_rigid = 50e-6
W_rigid = 200e-6
L_rigid = 500e-6
r_rigid = (H0_rigid**3 * W_rigid) / (12 * dynamic_viscosity * L_rigid) * 1e11

R0_tube_in = 150e-6
L_tube_in = 150000e-6
r_tube_in = (np.pi * R0_tube_in**4) / (8 * dynamic_viscosity * L_tube_in) * 1e11

R0_tube_out = 150e-6
L_tube_out = 150000e-6
r_tube_out = (np.pi * R0_tube_in**4) / (8 * dynamic_viscosity * L_tube_out) * 1e11

H0_bend = 200e-6
W_bend = 200e-6
L_bend = 1000e-6
r_bend = (H0_bend**3 * W_bend) / (12 * dynamic_viscosity * L_bend) * 1e11

laws = channel_law_rigid(
    r_rigid,
    r_tube_in,
    r_tube_out,
    r_bend,
)

outlet_velocity_network_for_different_inlet_pressure = []
different_inlet_pressure = []
ESR_array = []
outlet_Q_ratio_array = []
elasticity_coefficient_array = []


@pytest.mark.parametrize(
    (
        "inlet_pressure",
        "outlet_pressure",
        "elasticity_coefficient",
    ),
    [
        (0, 0, 1),
        (20, 0, 1),
        (40, 0, 1),
        (60, 0, 1),
        (80, 0, 1),
        (100, 0, 1),
        (120, 0, 1),
        (200, 0, 1),
        (400, 0, 1),
        (600, 0, 1),
        (800, 0, 1),
    ],
)
def test_flow_network_pressure_bcs(
    inlet_pressure,
    outlet_pressure,
    elasticity_coefficient,
):
    nodes, connectivity_ci = flat_top_hexagonal_microfluidic_network(
        1,
        6,
        7,
        center=True,
    )

    boundary_nodes = np.array([105, 107, 109, 111])
    boundary_pressure_b = np.array(
        [inlet_pressure, outlet_pressure, outlet_pressure, outlet_pressure]
    )

    # -------------------------------------------------------------------------
    # solve for the node pressures
    # -------------------------------------------------------------------------
    (
        pressure_nodes,
        flow_channels,
        conductance_channels,
        ESR,
        outlet_velocity_network,
        outlet_Q_ratio,
    ) = flow_network_1d_pressure_boundary_condition(
        connectivity_ci=connectivity_ci,
        boundary_nodes=boundary_nodes,
        boundary_pressures=boundary_pressure_b,
        channel_laws=laws,
        R0_tube_out=R0_tube_out,
        elasticity_coefficient=elasticity_coefficient,
    )
    outlet_velocity_network_for_different_inlet_pressure.append(outlet_velocity_network)
    different_inlet_pressure.append(inlet_pressure)
    ESR_array.append(ESR)
    outlet_Q_ratio_array.append(outlet_Q_ratio)
    elasticity_coefficient_array.append(elasticity_coefficient)

    print(outlet_Q_ratio)

    #print(outlet_velocity_network)

    #if inlet_pressure == 800:
    #    plot_flow_network(flow_channels, nodes, connectivity_ci)

    #if inlet_pressure == 800:
    #    plot_pressure_nodes_network(pressure_nodes, nodes, connectivity_ci)

    #if inlet_pressure == 800:
    #    plot_p_bar_network(pressure_nodes, nodes, connectivity_ci)

    #if inlet_pressure == 800:
    #    plot_conductance_network(conductance_channels, nodes, connectivity_ci[:-12],laws)

    if inlet_pressure == 800:
        plot_velocity_pressure_network(
            outlet_velocity_network_for_different_inlet_pressure,
            different_inlet_pressure,
            experimental_case="rigid", # None, "rigid", "centre", "shunt"
        )

    #if elasticity_coefficient == 4:
    #    plot_esr_outlet_flow_rate_ratio(ESR_array,outlet_Q_ratio_array, elasticity_coefficient_array)
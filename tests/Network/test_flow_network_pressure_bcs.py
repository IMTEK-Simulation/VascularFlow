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

outlet_velocity_network_for_different_inlet_pressure = []
different_inlet_pressure = []
ESR_array = []
outlet_Q_ratio_array = []
elasticity_coefficient_array = []


#params = [
#    (
#        800,
#        0,
#        np.random.uniform(30e-6, 70e-6),
#        np.random.uniform(1.0, 5.0),
#        np.random.uniform(100e-6, 500e-6),
#    )
#    for _ in range(200)
#]

params = [
    (0   , 0, 50e-6, 1, 250e-6),
    (20  , 0, 50e-6, 1, 250e-6),
    (40  , 0, 50e-6, 1, 250e-6),
    (60  , 0, 50e-6, 1, 250e-6),
    (80  , 0, 50e-6, 1, 250e-6),
    (100 , 0, 50e-6, 1, 250e-6),
    (120 , 0, 50e-6, 1, 250e-6),
    (200 , 0, 50e-6, 1, 250e-6),
    (400 , 0, 50e-6, 1, 250e-6),
    (600 , 0, 50e-6, 1, 250e-6),
    (800 , 0, 50e-6, 1, 250e-6),
]


@pytest.mark.parametrize(
    "inlet_pressure, outlet_pressure, H0_rigid, elasticity_coefficient, R0_tube",
    params
)

def test_flow_network_pressure_bcs(
    inlet_pressure,
    outlet_pressure,
    H0_rigid,
    elasticity_coefficient,
    R0_tube,
):
    print(
        f"\nparams: {inlet_pressure}, {outlet_pressure}, "
        f"{H0_rigid / 1e-6:.2f}e-6, "
        f"{elasticity_coefficient:.2f}, "
        f"{R0_tube / 1e-6:.2f}e-6"
    )


    dynamic_viscosity = 1e-3
    # Network channels dimensions
    W_rigid = 200e-6
    L_rigid = 500e-6
    r_rigid = (H0_rigid ** 3 * W_rigid) / (12 * dynamic_viscosity * L_rigid) * 1e11

    L_tube_in = 150000e-6
    r_tube_in = (np.pi * R0_tube ** 4) / (8 * dynamic_viscosity * L_tube_in) * 1e11

    L_tube_out = 150000e-6
    r_tube_out = (np.pi * R0_tube ** 4) / (8 * dynamic_viscosity * L_tube_out) * 1e11

    H0_bend = 200e-6
    W_bend = 200e-6
    L_bend = 1000e-6
    r_bend = (H0_bend ** 3 * W_bend) / (12 * dynamic_viscosity * L_bend) * 1e11

    laws = channel_law_center(
        r_rigid,
        r_tube_in,
        r_tube_out,
        r_bend,
    )

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
        R0_tube_out=R0_tube,
        elasticity_coefficient=elasticity_coefficient,
    )
    outlet_velocity_network_for_different_inlet_pressure.append(outlet_velocity_network)
    different_inlet_pressure.append(inlet_pressure)
    ESR_array.append(ESR)
    outlet_Q_ratio_array.append(outlet_Q_ratio)
    elasticity_coefficient_array.append(elasticity_coefficient)

    #print(f"outlet_Q_ratio: {outlet_Q_ratio}")

    #print(outlet_velocity_network)

    #if inlet_pressure == 800:
    #    plot_flow_network(flow_channels, nodes, connectivity_ci)

    if inlet_pressure == 800:
        plot_pressure_nodes_network(pressure_nodes, nodes, connectivity_ci)

    #if inlet_pressure == 800:
    #    plot_p_bar_network(pressure_nodes, nodes, connectivity_ci)

    #if inlet_pressure == 800:
    #    plot_conductance_network(conductance_channels, nodes, connectivity_ci[:-12],laws)

    #if inlet_pressure == 800:
    #    plot_velocity_pressure_network(
    #        outlet_velocity_network_for_different_inlet_pressure,
    #        different_inlet_pressure,
    #        experimental_case="centre", # None, "rigid", "centre", "shunt"
    #    )

    #if elasticity_coefficient == 4:
    #    plot_esr_outlet_flow_rate_ratio(ESR_array,outlet_Q_ratio_array, elasticity_coefficient_array)
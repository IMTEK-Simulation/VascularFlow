import numpy as np
from scipy.sparse import coo_array

from VascularFlow.Network.NewtonSolver import newton



def flow_network_1d_pressure_boundary_condition(
    connectivity_ci: np.ndarray,
    boundary_nodes: np.ndarray,
    boundary_pressures: np.ndarray,
    channel_laws,
    R0_tube_out,
    elasticity_coefficient,
):
    """
    Solve a 1D flow network with prescribed pressure boundary conditions.

    Parameters
    ----------
    connectivity_ci : np.ndarray of shape (n_channels, 2)
        Each row contains [inlet_node, outlet_node] for one channel.
    boundary_nodes : np.ndarray of shape (n_boundary,)
        Indices of nodes with fixed pressure.
    boundary_pressures : np.ndarray of shape (n_boundary,)
        Pressure values prescribed at boundary_nodes.
    channel_laws : list[dict]
        One law per channel. Supported forms:

        {"type": "rigid", "R": value}
        {"type": "toponly"}
        {"type": "topbottom"}
    R0_tube_out : float
        The cross-sectional radius of the network's outlet pipes
    elasticity_coefficient: float
        The coefficient of the constant parameters in the equation G(p_bar) that controls the intensity of
        the elasticity of the channels with the elastic wall

    Returns
    -------
    np.ndarray
        Solved nodal pressures.
    """

    # Extract inlet and outlet indices
    inlet_c, outlet_c = np.transpose(connectivity_ci)
    # Extract number of channels in the network
    nb_channels = len(inlet_c)
    if len(channel_laws) != nb_channels:
        raise ValueError(
            f"len(channel_laws) must equal number of channels ({nb_channels}), "
            f"but got {len(channel_laws)}"
        )
    # Extract number of nodes in the network
    nb_nodes = max(np.max(inlet_c), np.max(outlet_c)) + 1

    # Mark interior nodes (all except boundary)
    interior_mask = np.ones(nb_nodes, dtype=bool)
    interior_mask[boundary_nodes] = False
    # print(f"System has {nb_nodes} total nodes and {nb_nodes-len(boundary_b)} interior nodes.")

    # Initial pressure guess
    pressure_guess_n = np.linspace(
        boundary_pressures.min(),
        boundary_pressures.max(),
        nb_nodes,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Empirical channel-law parameters
    # Q = G(p_bar) Δp → volumetric flow rate (μl/s) and pressure (mbar) relationship in an elastic channel
    # Note:
    #   - tb means top and bottom walls are elastic
    #   - to means top wall is elastic
    #   - p_bar = (channel inlet pressure + channel outlet pressure) / 2
    # G(p_bar) = 1 / 1/f1 + 1 / f2
    # f1 = a + b * p_bar & f2 = c + d * p_bar
    # ------------------------------------------------------------------
    a_tb = 5.40344025e-01 * elasticity_coefficient
    b_tb = 1.00000000e-02 * elasticity_coefficient
    c_tb = 9.71964960e-01 * elasticity_coefficient
    d_tb = 4.12670568e-19 * elasticity_coefficient

    a_to = 5.59733101e-01 * elasticity_coefficient
    b_to = 7.11863699e-03 * elasticity_coefficient
    c_to = 8.31721642e-01 * elasticity_coefficient
    d_to = 5.56812759e-12 * elasticity_coefficient

    # ============================================================
    # Definition of G(p_bar) and its derivatives in tb and to channels
    # ============================================================
    def g_tb(pbar):
        f1 = a_tb + b_tb * pbar
        f2 = c_tb + d_tb * pbar
        return 1.0 / (1.0 / f1 + 1.0 / f2)

    def dg_tb_dpbar(pbar):
        f1 = a_tb + b_tb * pbar
        f2 = c_tb + d_tb * pbar
        num = b_tb / (f1**2) + d_tb / (f2**2)
        den = (1.0 / f1 + 1.0 / f2) ** 2
        return num / den

    def g_to(pbar):
        f1 = a_to + b_to * pbar
        f2 = c_to + d_to * pbar
        return 1.0 / (1.0 / f1 + 1.0 / f2)

    def dg_to_dpbar(pbar):
        f1 = a_to + b_to * pbar
        f2 = c_to + d_to * pbar
        num = b_to / (f1**2) + d_to / (f2**2)
        den = (1.0 / f1 + 1.0 / f2) ** 2
        return num / den

    # ------------------------------------------------------------------
    # Channel flow model and partial derivatives
    # ------------------------------------------------------------------
    def channel_flow_and_partials(inlet_pressure_c, outlet_pressure_c):
        dp_c = inlet_pressure_c - outlet_pressure_c
        pbar_c = 0.5 * (inlet_pressure_c + outlet_pressure_c)

        flow_c = np.zeros(nb_channels, dtype=float)
        dflow_dinlet_c = np.zeros(nb_channels, dtype=float)
        dflow_doutlet_c = np.zeros(nb_channels, dtype=float)

        for i, law in enumerate(channel_laws):
            law_type = law.get("type", "rigid").lower()
            dp = dp_c[i]
            pbar = pbar_c[i]
            if law_type == "rigid":
                R = float(law.get("R", 1.0))
                flow_c[i] = R * dp
                dflow_dinlet_c[i] = R
                dflow_doutlet_c[i] = -R

            elif law_type == "topbottom":
                G = g_tb(pbar)
                dG_dpbar = dg_tb_dpbar(pbar)
                flow_c[i] = G * dp
                dflow_dinlet_c[i] = 0.5 * dG_dpbar * dp + G
                dflow_doutlet_c[i] = 0.5 * dG_dpbar * dp - G

            elif law_type == "toponly":
                G = g_to(pbar)
                dG_dpbar = dg_to_dpbar(pbar)

                flow_c[i] = G * dp
                dflow_dinlet_c[i] = 0.5 * dG_dpbar * dp + G
                dflow_doutlet_c[i] = 0.5 * dG_dpbar * dp - G

        return flow_c, dflow_dinlet_c, dflow_doutlet_c

    # --------------------------------------------------------------------------
    # Node flow residuals: must equal zero at interior nodes
    # --------------------------------------------------------------------------
    def node_flow(pressure_n):
        inlet_pressure_c = pressure_n[inlet_c]
        outlet_pressure_c = pressure_n[outlet_c]
        flow_c, _, _ = channel_flow_and_partials(inlet_pressure_c, outlet_pressure_c)
        flow_n = -np.bincount(
            inlet_c, weights=flow_c, minlength=nb_nodes
        ) + np.bincount(outlet_c, weights=flow_c, minlength=nb_nodes)
        return flow_n

    # --------------------------------------------------------------------------
    # Global residual including pressure constraints at boundaries
    # --------------------------------------------------------------------------
    def residual(pressure_n):
        residual_n = node_flow(pressure_n)
        residual_n[boundary_nodes] = pressure_n[boundary_nodes] - boundary_pressures
        return residual_n

    # --------------------------------------------------------------------------
    # Jacobian Assembly
    # --------------------------------------------------------------------------
    def dnode_flow_dpressure(pressure_n):
        inlet_pressure_c = pressure_n[inlet_c]
        outlet_pressure_c = pressure_n[outlet_c]
        flow_c, dflow_dinlet_c, dflow_doutlet_c = channel_flow_and_partials(inlet_pressure_c, outlet_pressure_c)

        jac_nn = (
            coo_array(
                (dflow_dinlet_c, (outlet_c, inlet_c)),
                shape=(len(pressure_n), len(pressure_n)),
            )
            - coo_array(
                (dflow_doutlet_c, (inlet_c, outlet_c)),
                shape=(len(pressure_n), len(pressure_n)),
            )
            - coo_array(
                (dflow_dinlet_c, (inlet_c, inlet_c)),
                shape=(len(pressure_n), len(pressure_n)),
            )
            + coo_array(
                (dflow_doutlet_c, (outlet_c, outlet_c)),
                shape=(len(pressure_n), len(pressure_n)),
            )
        )
        return jac_nn

    # --------------------------------------------------------------------------
    # Global Jacobian with BC row enforcement
    # --------------------------------------------------------------------------
    def jacobian(pressure_n):
        coo_jac_nn = dnode_flow_dpressure(pressure_n)
        csr_jac_nn = coo_jac_nn.tocsr()
        csr_jac_nn[boundary_nodes] = 0
        csr_jac_nn[boundary_nodes, boundary_nodes] = 1
        return csr_jac_nn

    # Progress callback for debugging / convergence info
    def callback(iter_num: int, x: np.ndarray, f: np.ndarray, j) -> None:
        _ = x  # explicitly unused
        _ = j  # explicitly unused

        if f is None:
            print(f"Iteration {iter_num}: Initial guess assigned.")
            return

        residual_norm = np.linalg.norm(f, ord=np.inf)
        print(f"Iteration {iter_num}: ||residual||∞ = {residual_norm:.3e}")

    # Solve the system
    pressure_nodes = newton(
        fun=residual,
        x0=pressure_guess_n,
        jac=jacobian,
        callback=callback,
    )
    pressure_nodes = np.asarray(pressure_nodes, dtype=float)  # ensure consistent return type
    print("Pressure solution computed successfully.")

    inlet_pressure_channel = pressure_nodes[inlet_c]
    outlet_pressure_channel = pressure_nodes[outlet_c]
    flow_channels, _, _ = channel_flow_and_partials(inlet_pressure_channel, outlet_pressure_channel)

    Conductance_channels = np.zeros(nb_channels)
    Conductance_elastic_network = []
    Conductance_rigid_network = []

    for c, (i, j) in enumerate(connectivity_ci):
        pbar = 0.5 * (pressure_nodes[i] + pressure_nodes[j])
        law_type = channel_laws[c]["type"].lower()

        if law_type == "rigid":
            Conductance_channels[c] = float(channel_laws[c]["R"])
            Conductance_rigid_network.append(Conductance_channels[c])
        elif law_type == "toponly":
            Conductance_channels[c] = g_to(pbar)
            Conductance_elastic_network.append(Conductance_channels[c])
        elif law_type == "topbottom":
            Conductance_channels[c] = g_tb(pbar)
            Conductance_elastic_network.append(Conductance_channels[c])
        else:
            Conductance_channels[c] = np.nan

    Conductance_elastic_network_mean = np.mean(Conductance_elastic_network)
    Conductance_rigid_network_mean = np.mean(Conductance_rigid_network[:-12])
    # Elastic Strength Ratio calculation
    ESR = Conductance_elastic_network_mean / Conductance_rigid_network_mean

    #print("G_elastic_mean =", Conductance_elastic_network_mean)
    #print("G_rigid_mean =", Conductance_rigid_network_mean)
    #print("ESR =", ESR)

    # outlet velocity
    outlet_tube_cross_section = np.pi * R0_tube_out**2
    CI  = (flow_channels[-12] * 1e-9) / outlet_tube_cross_section
    CO  = (flow_channels[-4]  * 1e-9) / outlet_tube_cross_section
    SO1 = (flow_channels[-1]  * 1e-9) / outlet_tube_cross_section
    SO2 = (flow_channels[-7]  * 1e-9) / outlet_tube_cross_section

    outlet_velocity_network = [CI, CO, SO1, SO2]

    outlet_Q_ratio = flow_channels[-4] / flow_channels[-1]

    return pressure_nodes, flow_channels, Conductance_channels, ESR, outlet_velocity_network, outlet_Q_ratio

import numpy as np
from scipy.sparse import coo_array, csr_array



from VascularFlow.Network.NewtonSolver import newton


def flow_network_1d_pressure_boundary_condition(
    connectivity_ci: np.ndarray,
    boundary_nodes: np.ndarray,
    boundary_pressures: np.ndarray,
    resistance: float,
) -> np.ndarray:
    """
    Solve fluid pressure distribution in a 1D network with linear or nonlinear pressure-flow relation.

    Parameters
    ----------
    connectivity_ci : np.ndarray of shape (C, 2)
        Directed connectivity array where each row (i, j) indicates a channel from
        inlet node i to outlet node j.
    boundary_nodes : np.ndarray of shape (B,)
        Node indices where pressure is prescribed (Dirichlet boundary nodes).
    boundary_pressures : np.ndarray of shape (B,)
        Corresponding fixed pressure values for each boundary node.
    resistance : float
        Hydraulic resistance of every channel.

    Returns
    -------
    np.ndarray
        Pressure solution for all nodes in the network.

    Notes
    -----
    This function:
    * Forms a linear or nonlinear system enforcing mass conservation at interior nodes.
    * Enforces fixed pressure boundary conditions directly in the residual.
    * Solves via Newton iteration (1 step for system is linear).
    """
    # Extract inlet and outlet indices
    inlet_id, outlet_id = np.transpose(connectivity_ci)

    # Determine number of nodes
    nb_nodes = max(np.max(inlet_id), np.max(outlet_id)) + 1
    print(f"Network size: {nb_nodes} nodes, {len(boundary_nodes)} fixed boundaries.")

    # Mark interior nodes (all except boundary)
    interior_mask = np.ones(nb_nodes, dtype=bool)
    interior_mask[boundary_nodes] = False

    # Initial pressure guess
    pressure = np.arange(nb_nodes, dtype=float)

    def channel_flow(pin: np.ndarray, pout: np.ndarray) -> np.ndarray:
        #return resistance * (pin - pout)
        return resistance * (pin - pout) ** 2

    def dq_dpin(pin: np.ndarray, pout: np.ndarray) -> np.ndarray:
        # ∂Q/∂p_in = R
        #return resistance * np.ones_like(pin)
        return 2 * resistance * (pin - pout)

    def dq_dpout(pin: np.ndarray, pout: np.ndarray) -> np.ndarray:
        # ∂Q/∂p_out = -R
        #return -resistance * np.ones_like(pout)
        return - 2 * resistance * (pin - pout)

    # --------------------------------------------------------------------------
    # Node flow residuals: must equal zero at interior nodes
    # --------------------------------------------------------------------------
    def node_flow(p_vec: np.ndarray) -> np.ndarray:
        pin = p_vec[inlet_id]
        pout = p_vec[outlet_id]
        flow = channel_flow(pin, pout)

        # Flow imbalance: inflow - outflow = 0
        return (
                -np.bincount(inlet_id, weights=flow, minlength=nb_nodes) +
                np.bincount(outlet_id, weights=flow, minlength=nb_nodes)
        )

    # --------------------------------------------------------------------------
    # Jacobian Assembly
    # --------------------------------------------------------------------------
    def dnode_flow_dpressure(p_vec: np.ndarray) -> csr_array:
        pin = p_vec[inlet_id]
        pout = p_vec[outlet_id]
        dQ_in = dq_dpin(pin, pout)
        dQ_out = dq_dpout(pin, pout)

        # Sparse assembly of Jacobian components
        jac = (
                coo_array((dQ_in, (outlet_id, inlet_id)), shape=(nb_nodes, nb_nodes)) -
                coo_array((dQ_out, (inlet_id, outlet_id)), shape=(nb_nodes, nb_nodes)) -
                coo_array((dQ_in, (inlet_id, inlet_id)), shape=(nb_nodes, nb_nodes)) +
                coo_array((dQ_out, (outlet_id, outlet_id)), shape=(nb_nodes, nb_nodes))
        ).tocsr()

        return jac

    # --------------------------------------------------------------------------
    # Global residual including pressure constraints at boundaries
    # --------------------------------------------------------------------------
    def residual(p_vec: np.ndarray) -> np.ndarray:
        r = node_flow(p_vec)
        r[boundary_nodes] = p_vec[boundary_nodes] - boundary_pressures
        return r

    # --------------------------------------------------------------------------
    # Global Jacobian with BC row enforcement
    # --------------------------------------------------------------------------
    def jacobian(p_vec: np.ndarray) -> csr_array:
        j = dnode_flow_dpressure(p_vec)

        # Override BC rows
        j[boundary_nodes, :] = 0
        j[boundary_nodes, boundary_nodes] = 1

        return j

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
    pressure = newton(
        fun=residual,
        x0=pressure,
        jac=jacobian,
        callback=callback,
    )
    pressure = np.asarray(pressure, dtype=float)  # ensure consistent return type
    print("Pressure solution computed successfully.")
    return pressure

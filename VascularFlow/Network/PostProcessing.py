import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_flow_network(
    pressure_n: np.ndarray,
    inlet_c: np.ndarray,
    outlet_c: np.ndarray,
    nodes: np.ndarray | None = None,
    title: str = "Flow + Pressure Distribution",
) -> None:
    """
    Visualize pressure and flow on a 1D network.

    Parameters
    ----------
    pressure_n : np.ndarray
        Pressure at each node (length = number of nodes).
    inlet_c, outlet_c : np.ndarray
        Arrays of equal length describing channel connectivity:
        each pair (inlet_c[k], outlet_c[k]) is one channel.
    nodes : np.ndarray, optional
        Node coordinates (N, 2). If provided, used as positions.
        If None, a force-directed network layout is applied.
    title : str
        Plot title.
    """

    # ---------------------------------------------------------------
    # Build graph and attach pressure + flow attributes
    # ---------------------------------------------------------------
    G = nx.Graph()

    for n, p in enumerate(pressure_n):
        G.add_node(n, pressure=float(p))

    flow_c = pressure_n[inlet_c] - pressure_n[outlet_c]
    for i, o, f in zip(inlet_c, outlet_c, flow_c):
        G.add_edge(int(i), int(o), flow=float(f))

    # ---------------------------------------------------------------
    # Position: geometry if given, else spring layout
    # ---------------------------------------------------------------
    if nodes is not None:
        pos = {i: (nodes[i, 0], nodes[i, 1]) for i in range(len(nodes))}
    else:
        pos = nx.spring_layout(G, seed=42)

    # ---------------------------------------------------------------
    # Scale visual properties
    # ---------------------------------------------------------------
    p_min, p_max = pressure_n.min(), pressure_n.max()
    if p_max == p_min:
        p_max = p_min + 1.0  # avoid zero range if all pressures same

    # Scale node sizes by pressure
    node_sizes = 200 + 200 * (pressure_n - p_min) / (p_max - p_min)

    edges = list(G.edges(data=True))
    flows = np.array([abs(d["flow"]) for (_, _, d) in edges])
    max_flow = flows.max() if np.any(flows > 0) else 1.0

    edge_widths = 0.5 + 3.0 * (flows / max_flow)

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background edges (light)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="lightgray", width=0.8, alpha=0.6
    )

    # Flow-scaled dark edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, edge_color="black"
    )

    # Nodes colored by pressure
    nodes_plot = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=pressure_n,
        cmap="coolwarm",
        vmin=p_min,
        vmax=p_max,
    )

    # ---------------------------------------------------------------
    # Updated Labels: Pressure values instead of node numbers
    # ---------------------------------------------------------------
    labels = {n: f"{pressure_n[n]:.1f}" for n in range(len(pressure_n))}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8, font_color="black", font_weight="bold"
    )

    # Color bar for pressure
    cbar = plt.colorbar(nodes_plot)
    cbar.set_label("Pressure Value")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("Network visualization completed.")

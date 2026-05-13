import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ============================================================
# Global plotting style
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


def plot_flow_network(
    flow_channels,
    nodes,
    connectivity_ci,
):

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=50, color="royalblue")

    # plot channels + Q values
    for c, (i, j) in enumerate(connectivity_ci):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        # draw line
        ax.plot([x1, x2], [y1, y2], "gray", linewidth=flow_channels[c] / 5)

        # midpoint for text
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)

        # show Q
        #ax.text(
        #    xm, ym,
        #    f"{flow_channels[c]:.2f}",
        #    fontsize=6,
        #    color="black",
        #    ha="center",
        #    va="center"
        #)

    # labels
    #ax.set_title("Flow Distribution")
    ax.set_aspect("equal")
    #ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.savefig("flow_network.png")

    plt.show()

def plot_pressure_nodes_network(
    pressure_nodes,
    nodes,
    connectivity_ci,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(nodes[:, 0], nodes[:, 1], s=50, color="royalblue")

    for c, (i, j) in enumerate(connectivity_ci):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        ax.plot([x1, x2], [y1, y2], "gray", linewidth=1)

    for n, (x, y) in enumerate(nodes):
        ax.text(
            x, y,
            f"{pressure_nodes[n]:.1f}",
            fontsize=8,
            color="blue",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

    ax.set_title("Network with Nodal Pressures")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.show()

def plot_p_bar_network(
    pressure_nodes,
    nodes,
    connectivity_ci,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(nodes[:, 0], nodes[:, 1], s=50, color="royalblue")

    # draw branches + show P_bar on each branch
    for c, (i, j) in enumerate(connectivity_ci):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        ax.plot([x1, x2], [y1, y2], "gray", linewidth=1)

        pbar = 0.5 * (pressure_nodes[i] + pressure_nodes[j])
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)

        ax.text(
            xm, ym,
            f"{pbar:.1f}",
            fontsize=6,
            color="red",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

    ax.set_title("Network with Nodal Pressures and Branch P_bar")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.show()

def plot_conductance_network(
    Conductance_channels,
    nodes,
    connectivity_ci,
    channel_laws,
):
    G_c = np.asarray(Conductance_channels, dtype=float)

    G_min = np.nanmin(G_c)
    G_max = np.nanmax(G_c)

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=50, color="black")

    for c, (i, j) in enumerate(connectivity_ci):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        law_type = channel_laws[c]["type"].lower()

        # color by channel type
        if law_type == "rigid":
            color = "gray"
        elif law_type == "toponly":
            color = "blue"
        elif law_type == "topbottom":
            color = "red"
        else:
            color = "green"

        # width by G magnitude
        if G_max > G_min:
            G_norm = (G_c[c] - G_min) / (G_max - G_min)
            width = 0.8 + 200 * G_norm
        else:
            width = 1.5

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width)

        # midpoint
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)

        # show G
        ax.text(
            xm,
            ym - 0.08,
            f"{G_c[c]:.3f}",
            fontsize=5,
            color="darkgreen",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, label="Rigid"),
        Line2D([0], [0], color="blue", lw=2, label="TopOnly"),
        Line2D([0], [0], color="red", lw=2, label="TopBottom"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_title("Network: channel type + conductance G")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.show()


def plot_velocity_pressure_network(
    outlet_velocity_network_for_different_inlet_pressure,
    different_inlet_pressure,
    experimental_case=None,  # None, "rigid", "centre", "shunt"
):
    outlet_velocity_network_for_different_inlet_pressure = np.array(
        outlet_velocity_network_for_different_inlet_pressure
    )
    different_inlet_pressure = np.array(different_inlet_pressure)

    # ---------------- Simulation results ----------------
    CI_sim = outlet_velocity_network_for_different_inlet_pressure[:, 0]
    CO_sim = outlet_velocity_network_for_different_inlet_pressure[:, 1]
    SO1_sim = outlet_velocity_network_for_different_inlet_pressure[:, 2]
    SO2_sim = outlet_velocity_network_for_different_inlet_pressure[:, 3]

    # ---------------- Experimental results ----------------
    exp_data = {
        "rigid": {
            "CI": np.array([0, 0.0132, 0.0248, 0.0380, 0.0513, 0.0653, 0.0786, 0.1332, 0.2673, 0.4031, 0.5380]),
            "CO": np.array([0, 0.0037, 0.0078, 0.0120, 0.0161, 0.0202, 0.0244, 0.0401, 0.0815, 0.1237, 0.1659]),
            "SO1": np.array([0, 0.0045, 0.0086, 0.0144, 0.0211, 0.0260, 0.0310, 0.0509, 0.0972, 0.1461, 0.1941]),
            "SO2": np.array([0, 0.0037, 0.0095, 0.0144, 0.0211, 0.0252, 0.0293, 0.0467, 0.0923, 0.1361, 0.1792]),
        },
        "centre": {
            "CI": np.array([0, 0.008, 0.024, 0.037, 0.051, 0.066, 0.081, 0.145, 0.327, 0.550, 0.817]),
            "CO": np.array([0, 0.002, 0.007, 0.011, 0.016, 0.022, 0.027, 0.055, 0.143, 0.272, 0.446]),
            "SO1": np.array([0, 0.002, 0.007, 0.011, 0.016, 0.022, 0.027, 0.046, 0.097, 0.145, 0.194]),
            "SO2": np.array([0, 0.002, 0.007, 0.011, 0.016, 0.022, 0.027, 0.044, 0.087, 0.134, 0.178]),
        },
        "shunt": {
            "CI": np.array([0, 0.0147, 0.0339, 0.0487, 0.0649, 0.0856, 0.1048, 0.1816, 0.4002, 0.6587, 0.9601]),
            "CO": np.array([0, 0.0029, 0.0073, 0.0132, 0.0192, 0.0236, 0.0295, 0.0561, 0.1432, 0.2732, 0.4475]),
            "SO1": np.array([0, 0.0103, 0.0192, 0.0236, 0.0310, 0.0384, 0.0443, 0.0679, 0.1344, 0.2023, 0.2673]),
            "SO2": np.array([0, 0.0088, 0.0147, 0.0206, 0.0310, 0.0354, 0.0428, 0.0649, 0.1240, 0.1875, 0.2496]),
        },
    }

    plt.figure(figsize=(7.5, 4.5))

    # ---------------- Simulation plots ----------------
    plt.plot(
        different_inlet_pressure, CI_sim,
        marker='^', markersize=7,
        color='green', label='CI - Simulation'
    )

    plt.plot(
        different_inlet_pressure, SO1_sim,
        marker='s', markersize=6,
        linestyle='--', color='black',
        label=r'SO$_1$ - Simulation'
    )

    plt.plot(
        different_inlet_pressure, SO2_sim,
        marker='D', markersize=6,
        linestyle='--', color='blue',
        label=r'SO$_2$ - Simulation'
    )

    plt.plot(
        different_inlet_pressure, CO_sim,
        marker='o', markersize=7,
        color='orange', label='CO - Simulation'
    )

    # ---------------- Experimental plots ----------------
    if experimental_case is not None:
        if experimental_case not in exp_data:
            raise ValueError(
                "experimental_case must be one of: None, 'rigid', 'centre', 'shunt'"
            )

        CI_exp = exp_data[experimental_case]["CI"]
        CO_exp = exp_data[experimental_case]["CO"]
        SO1_exp = exp_data[experimental_case]["SO1"]
        SO2_exp = exp_data[experimental_case]["SO2"]

        plt.scatter(
            different_inlet_pressure, CI_exp,
            marker='^', s=65,
            facecolors='none', edgecolors='green',
            linewidths=1.8, label='CI - Experimental'
        )

        plt.scatter(
            different_inlet_pressure, SO1_exp,
            marker='s', s=65,
            facecolors='none', edgecolors='black',
            linewidths=1.8, label=r'SO$_1$ - Experimental'
        )

        plt.scatter(
            different_inlet_pressure, SO2_exp,
            marker='D', s=55,
            facecolors='none', edgecolors='blue',
            linewidths=1.8, label=r'SO$_2$ - Experimental'
        )

        plt.scatter(
            different_inlet_pressure, CO_exp,
            marker='o', s=55,
            facecolors='none', edgecolors='orange',
            linewidths=1.8, label='CO - Experimental'
        )

    #plt.title("Simulation vs Experimental")
    plt.xlabel(r'$\Delta P_{\mathrm{in}}$ (mbar)')
    plt.ylabel(r'Outlet velocity $u$ (m/s)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tick_params(direction='in', length=5, width=1, top=True, right=True)
    plt.legend(loc='upper left', frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig("network_validation.pdf")
    plt.show()

def plot_esr_outlet_flow_rate_ratio(
    ESR_array,
    outlet_Q_ratio_array,
    elasticity_coefficient_array,
):
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(ESR_array, outlet_Q_ratio_array, marker='^', linewidth=2.5, markersize=7, color='red')
    indices = [0, len(ESR_array) - 1]

    for i in indices:
        plt.text(
            ESR_array[i],
            outlet_Q_ratio_array[i],
            f"$\\alpha={elasticity_coefficient_array[i]:.1f}$",
            fontsize=12,
            ha='left',
            va='bottom'
        )

    plt.title("Effect of Elastic Strength on Flow Redistribution")
    plt.xlabel(r"Elastic Strength Ratio (ESR) = $\frac{\overline{G}_{elastic}}{\overline{G}_{rigid}}$")
    plt.ylabel(r"Flow Ratio $Q_{CO} / Q_{SO}$")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(direction='in', length=5)
    plt.tight_layout()
    plt.show()


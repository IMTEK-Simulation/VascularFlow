import matplotlib.pyplot as plt

from VascularFlow.Network.FlatTopHexagonalNetworkGeometry import flat_top_hexagonal_microfluidic_network


def test_hexagonal_network_geometry() -> None:
    """
    Test and visualize the hexagonal microfluidic network geometry.

    This function:
    1. Generates a hexagonal microfluidic network using the provided geometry function.
    2. Prints the connectivity list for verification.
    3. Plots all nodes in a Cartesian coordinate system.
    4. Overlays node indices and channels for visual confirmation.
    5. Creates and prints a dictionary mapping node IDs to coordinates.
    """

    # Generate network nodes and connectivity
    outer_radius = 1.0
    num_rows = 3
    num_cols = 5
    nodes, connectivity_ci = flat_top_hexagonal_microfluidic_network(
        outer_radius, num_rows, num_cols
    )

    print("Connectivity (inlet → outlet):")
    print(connectivity_ci)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(nodes[:, 0], nodes[:, 1], s=60, color="royalblue")

    ax.set_title("Hexagonal Microfluidic Network\n(Flat-top Orientation)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Plot channels as line segments for clearer visualization
    for inlet, outlet in connectivity_ci:
        x_coords = [nodes[inlet, 0], nodes[outlet, 0]]
        y_coords = [nodes[inlet, 1], nodes[outlet, 1]]
        ax.plot(x_coords, y_coords, "gray", linewidth=1)

    # Label node indices
    for node_id, (x, y) in enumerate(nodes):
        ax.text(x, y, str(node_id),
                fontsize=10, ha="center", va="center", color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6))

    plt.show()

    # ------------------------------------------------------------
    # Build and display node dictionary
    # ------------------------------------------------------------
    #node_dict = {i: tuple(coord) for i, coord in enumerate(nodes)}
    #print("\nNode Dictionary (ID → Coordinates):")
    #for k, v in node_dict.items():
    #    print(f"{k:2d}: {v}")

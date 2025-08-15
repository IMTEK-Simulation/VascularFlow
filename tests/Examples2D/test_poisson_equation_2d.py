import numpy as np
import pytest
import matplotlib.pyplot as plt

from VascularFlow.Examples2D.PoissonEquation import poisson_equation_solver
from VascularFlow.Numerics.BasisFunctions import BilinearShapeFunctions

def test_plot_poisson_solution_2d():
    # --- problem setup ---
    L = 1.0
    H = 1.0
    n_x = 9
    bc_positions = ["bottom", "right", "top", "left"]
    bc_values = [0.0, 0.0, 0.0, 0.0]
    source_func = lambda s, n: -6.0  # constant rhs

    shape_function = BilinearShapeFunctions()

    # --- solve ---
    T = poisson_equation_solver(
        shape_function=shape_function,
        domain_length=L,
        domain_height=H,
        n_x=n_x,
        bc_positions=bc_positions,
        bc_values=bc_values,
        source_func=source_func,
    )

    # --- reshape to grid ---
    dx = L / (n_x - 1)
    n_y = int(round(H / dx)) + 1
    T_grid = T.reshape(n_y, n_x)  # row-major: j (y) × i (x)

    # physical coordinates (for axis ticks)
    xs = np.linspace(0.0, L, n_x)
    ys = np.linspace(0.0, H, n_y)

    # --- 2D heatmap ---
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        T_grid,
        extent=[0, L, 0, H],
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )
    plt.colorbar(im, label="φ(x, y)")
    plt.title("Poisson solution (Q1, Dirichlet 0 on all sides, f = -6)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


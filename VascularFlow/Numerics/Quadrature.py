import numpy as np

_quadrature_points = {
    1: np.array([0]),
    2: np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]),
    3: np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]),
    4: np.array(
        [
            -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
            -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
        ]
    ),
    5: np.array(
        [
            -1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
            -1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
            0,
            1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
            1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
        ]
    ),
}

_quadrature_weights = {
    1: np.array([2]),
    2: np.array([1, 1]),
    3: np.array([5 / 9, 8 / 9, 5 / 9]),
    4: np.array(
        [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]
    ),
    5: np.array(
        [
            (322 - 13 * np.sqrt(70)) / 900,
            (322 + 13 * np.sqrt(70)) / 900,
            128 / 225,
            (322 + 13 * np.sqrt(70)) / 900,
            (322 - 13 * np.sqrt(70)) / 900,
        ]
    ),
}


def gaussian_quadrature(nb_quad_pts, func):
    """
    Compute the integral of a function using Gaussian quadrature
    over the interval [-1, 1]
    """
    points = _quadrature_points[nb_quad_pts]
    weights = _quadrature_weights[nb_quad_pts]
    return np.sum(weights * func(points))

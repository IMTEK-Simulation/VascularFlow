import numpy as np

########################################################## 1D ##########################################################

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


def gaussian_quadrature1(nb_quad_pts: int, func: callable) -> float:
    """
    Compute the integral of a function using Gaussian quadrature
    over the interval [-1, 1]
    """
    points = _quadrature_points[nb_quad_pts]
    weights = _quadrature_weights[nb_quad_pts]
    return np.sum(weights * func(points), axis=-1)


def gaussian_quadrature(nb_quad_pts: int, a: float, b: float, func: callable) -> float:
    """
    Compute the integral of a function using Gaussian quadrature
    over the interval [a, b]
    """
    return (
        0.5
        * (b - a)
        * gaussian_quadrature1(
            nb_quad_pts, lambda x: func(0.5 * (b - a) * x + 0.5 * (b + a))
        )
    )


########################################################## 2D ##########################################################

_quadrature_points_2D = {
    4: np.array(
        [
            [-np.sqrt(1 / 3), -np.sqrt(1 / 3)],
            [np.sqrt(1 / 3), -np.sqrt(1 / 3)],
            [np.sqrt(1 / 3), np.sqrt(1 / 3)],
            [-np.sqrt(1 / 3), np.sqrt(1 / 3)],
        ]
    ),
    9: np.array(
        [
            [0,                         0],
            [-np.sqrt(0.6), -np.sqrt(0.6)],
            [-np.sqrt(0.6),  np.sqrt(0.6)],
            [np.sqrt(0.6),   np.sqrt(0.6)],
            [np.sqrt(0.6),  -np.sqrt(0.6)],
            [-np.sqrt(0.6),             0],
            [0,              np.sqrt(0.6)],
            [np.sqrt(0.6),              0],
            [0,             -np.sqrt(0.6)],
        ]
    ),
}

_quadrature_weights_2D = {
    4: np.array(
        [
            1,
            1,
            1,
            1,
        ]
    ),
    9: np.array(
        [
            (8 / 9) ** 2,
            (5 / 9) ** 2,
            (5 / 9) ** 2,
            (5 / 9) ** 2,
            (5 / 9) ** 2,
            (5 / 9) * (8 / 9),
            (5 / 9) * (8 / 9),
            (5 / 9) * (8 / 9),
            (5 / 9) * (8 / 9),
        ]
    ),
}



def integrate_over_square(func: callable, nb_quad_pts_2d: int) -> float:
    """
    Integrate a 2D function f(ξ, η) over the reference square [-1, 1] × [-1, 1]
    using 2D Gaussian quadrature.

    Parameters
    ----------
    func : callable
        A scalar function f(ξ, η) to integrate. Should accept two float arguments (ξ, η).
    nb_quad_pts_2d : int, optional
        Number of quadrature points (supported: 4 or 9). Default is 9.

    Returns
    -------
    float
        Approximate integral value using 2D Gaussian quadrature.
    """
    gauss_points = _quadrature_points_2D[nb_quad_pts_2d]
    weights = _quadrature_weights_2D[nb_quad_pts_2d]

    integral = 0.0
    for (s, n), w in zip(gauss_points, weights):
        integral += w * func(s, n)

    return integral
